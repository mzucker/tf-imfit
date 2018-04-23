from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image

ModelsTuple = namedtuple('ModelsTuple',
                         'full, local, preview')

InputsTuple = namedtuple('InputsTuple',
                         'input_image, weight_image, '
                         'x, y, target_tensor, max_row')

StateTuple = namedtuple('StateTuple', 'params, gabor, con_loss')

######################################################################
# Indexes of Gabor function parameters

GABOR_PARAM_U = 0 # [-1, 1]
GABOR_PARAM_V = 1 # [-1, 1]
GABOR_PARAM_R = 2 # [0, 2*pi]
GABOR_PARAM_P = 3 # [0, 2*pi]
GABOR_PARAM_L = 4 # [2.5*px, 4]
GABOR_PARAM_T = 5 # [px, 4]
GABOR_PARAM_S = 6 # [px, 2]
GABOR_PARAM_H = 7 # [0, 2]

GABOR_NUM_PARAMS = 8

GABOR_RANGE = np.array([
    [ -1, 1 ],
    [ -1, 1 ],
    [ -np.pi, np.pi ],
    [ -np.pi, np.pi ],
    [ 0, 4 ],
    [ 0, 4 ],
    [ 0, 2 ],
    [ 0, 2 ] ])

######################################################################
# Parse a duration string

def parse_duration(dstr):

    expr = r'^(([0-9]+)(:|h))?([0-9]+)((:|m)(([0-9]+)s?))?$'

    g = re.match(expr, dstr)

    if g is None:
        raise argparse.ArgumentTypeError(dstr + ': invalid duration format')

    def make_int(x):
        if x is None:
            return 0
        else:
            return int(x)

    h = make_int( g.group(2) )
    m = make_int( g.group(4) )
    s = make_int( g.group(8) )

    return (h*60 + m)*60 + s

######################################################################
# Parse command-line options, return namespace containing results

def get_options():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('image', type=argparse.FileType('r'),
                        metavar='IMAGE.png',
                        help='image to approximate')

    parser.add_argument('-s', '--max-size', type=int, metavar='N',
                        help='maximum size of image to load',
                        default=128)

    parser.add_argument('-p', '--preview-size', type=int, metavar='N',
                        default=0,
                        help='size of preview image (<0 to disable)')
    
    parser.add_argument('-w', '--weights', type=argparse.FileType('r'),
                        metavar='WEIGHTS.png',
                        help='load weights from file',
                        default=None)

    parser.add_argument('-i', '--input', type=str,
                        metavar='PARAMFILE.txt',
                        help='read input params from file')

    parser.add_argument('-o', '--output', type=str,
                        metavar='PARAMFILE.txt',
                        help='write input params to file')
    
    parser.add_argument('-t', '--time-limit', type=parse_duration,
                        metavar='LIMIT',
                        help='time limit (e.g. 1:30 or 1h30m)',
                        default=None)

    parser.add_argument('-T', '--total-iterations', type=int,
                        metavar='N',
                        help='total limit on outer loop iterations',
                        default=None)

    parser.add_argument('-n', '--num-models', type=int, metavar='N',
                        help='total number of models to fit',
                        default=128)
 
    parser.add_argument('-e', '--mini-ensemble-size', type=int, metavar='N',
                        help='models to update per local fit',
                        default=1)
   
    parser.add_argument('-L', '--num-local', type=int, metavar='N',
                        help='number of random guesses per local fit',
                        default=200)

    parser.add_argument('-l', '--local-iter', type=int, metavar='N',
                        help='maximum # of iterations per local fit',
                        default=100)
    
    parser.add_argument('-F', '--full-every', type=int, metavar='N', default=32,
                        help='perform joint optimization after every N outer loops')

    parser.add_argument('-f', '--full-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=10000)
    
    parser.add_argument('-r', '--local-learning-rate', type=float, metavar='R',
                        help='learning rate for local opt.',
                        default=0.01)

    parser.add_argument('-R', '--full-learning-rate', type=float, metavar='R',
                        help='learning rate for full opt.',
                        default=0.001)

    parser.add_argument('-P', '--perturb-amount', type=float,
                        metavar='R', default=0.01,
                        help='amount to perturb replacement fits by')

    parser.add_argument('-c', '--copy-quantity', type=float,
                        metavar='C',
                        help='number or fraction of re-fits to initialize with cur. model',
                        default=0.5)
    
    parser.add_argument('-B', '--lambda-err', type=float,
                        metavar='LAMBDA',
                        help='weight on Boltzmann sampling of error map',
                        default=2.0)

    parser.add_argument('-a', '--anneal-temp', type=float, metavar='T',
                        help='temperature for simulated annealing',
                        default=0.0)
        
    parser.add_argument('-S', '--label-snapshot', action='store_true',
                        help='individually label snapshots (good for anim. gif)')
 
    parser.add_argument('-x', '--snapshot-prefix', type=str, metavar='BASENAME',
                        help='prefix for snapshots', default='out')
  
    opts = parser.parse_args()

    if opts.copy_quantity < 0:
        opts.copy_quantity = 0
    elif opts.copy_quantity >= 1:
        opts.copy_quantity = 1
    else:
        opts.copy_quantity = int(round(opts.copy_quantity * opts.num_local))

    if opts.preview_size == 0:
        opts.preview_size = 4*opts.max_size
    elif opts.preview_size < 0:
        opts.preview_size = 0

    assert opts.num_models % opts.mini_ensemble_size == 0
    
    return opts

######################################################################
# Proportional scaling for image shape

def scale_shape(shape, desired_size):

    h, w = shape

    if w > h:
        wnew = desired_size
        hnew = int(round(float(h) * desired_size / w))
    else:
        wnew = int(round(float(w) * desired_size / h))
        hnew = desired_size

    return hnew, wnew

######################################################################
# Open an image, convert it to grayscale, resize to desired size
        
def open_grayscale(handle, max_size):

    if handle is None:
        return None

    image = Image.open(handle)

    if image.mode != 'L':
        print('converting {} to grayscale'.format(handle.name))
        image = image.convert('L')

    w, h = image.size
    
    if max(w, h) > max_size:
        h, w = scale_shape((h, w), max_size)
        image = image.resize((w, h), resample=Image.LANCZOS)

    print('{} is {}x{}'.format(handle.name, w, h))

    image = np.array(image).astype(np.float32) / 255.
    
    return image

######################################################################

def mix(a, b, u):
    return a + u*(b-a)

######################################################################
# Encapsulate the tensorflow objects we need to run our fit.
# Note we will create several of these (see main function below).

class GaborModel(object):

    def __init__(self, x, y, minishape, weight,
                 target,
                 learning_rate=0.0001,
                 params=None,
                 initializer=None,
                 max_row=None):

        # The Gabor function tensor we define will be f x m x h x w
        # where f is the number of parallel, independent fits being computed,
        # and m is the number of models per ensemble
        num_parallel, ensemble_size = minishape

        # Allow evaluating less than ensemble_size models (i.e. while
        # building up full model).
        if max_row is None:
            max_row = ensemble_size

        gmin = GABOR_RANGE[:,0].reshape(1,1,GABOR_NUM_PARAMS).copy()
        gmax = GABOR_RANGE[:,1].reshape(1,1,GABOR_NUM_PARAMS).copy()
            
        # Parameter tensor could be passed in or created here
        if params is not None:

            self.params = params

        else:

            if initializer is None:
                initializer = tf.random_uniform_initializer(minval=gmin,
                                                            maxval=gmax,
                                                            dtype=tf.float32)

            # f x m x 8
            self.params = tf.get_variable(
                'params',
                shape=(num_parallel, ensemble_size, GABOR_NUM_PARAMS),
                dtype=tf.float32,
                initializer=initializer)

        gmin[:,:,:GABOR_PARAM_L] = -np.inf
        gmax[:,:,:GABOR_PARAM_L] =  np.inf

        self.cparams = tf.clip_by_value(self.params[:,:max_row],
                                        gmin, gmax,
                                        name='cparams')

        ############################################################
        # Now compute the Gabor function for each fit/model
        
        # f x m x 8 x 1 x 1
        params_bcast = self.cparams[:,:,:,None,None]

        # f x m x 1 x 1
        u = params_bcast[:,:,GABOR_PARAM_U]
        v = params_bcast[:,:,GABOR_PARAM_V]
        r = params_bcast[:,:,GABOR_PARAM_R]
        p = params_bcast[:,:,GABOR_PARAM_P]
        l = params_bcast[:,:,GABOR_PARAM_L]
        s = params_bcast[:,:,GABOR_PARAM_S]
        t = params_bcast[:,:,GABOR_PARAM_T]
        h = params_bcast[:,:,GABOR_PARAM_H]

        cr = tf.cos(r)
        sr = tf.sin(r)

        f = np.float32(2*np.pi) / l

        s2 = s*s
        t2 = t*t

        # f x m x 1 x w
        xp = x-u

        # f x m x h x 1
        yp = y-v

        # f x m x h x w
        b1 =  cr*xp + sr*yp
        b2 = -sr*xp + cr*yp

        b12 = b1*b1
        b22 = b2*b2

        w = tf.exp(-b12/(2*s2) - b22/(2*t2))

        k = f*b1 +  p
        ck = tf.cos(k)

        self.gabor = tf.identity(h * w * ck, name='gabor')

        ############################################################
        # Compute the ensemble sum of all models for each fit        
        
        # f x h x w
        self.approx = tf.reduce_sum(self.gabor, axis=1, name='approx')

        ############################################################
        # Everything below here is for optimizing, if we just want
        # to visualize, stop now.
        
        if target is None:
            return

        ############################################################
        # Compute loss for soft constraints
        #
        # All constraint losses are of the form min(c, 0)**2, where c
        # is an individual constraint function. So we only get a
        # penalty if the constraint function c is less than zero.

        # Box constraints on l,t,s,h. Note we don't actually
        # enforce bounds on u,v,r,p


        box_constraints = []

        for i  in range(GABOR_PARAM_P, GABOR_NUM_PARAMS):
            lo, hi = GABOR_RANGE[i]
            var = self.cparams[:,:,i]
            if lo != -np.pi:
                box_constraints.append( var - lo )
            if hi != np.pi:
                box_constraints.append( hi - var )
        
        # Pair-wise constraints on l, s, t:

        # f x m x 1
        l = self.params[:,:,GABOR_PARAM_L]
        s = self.params[:,:,GABOR_PARAM_S]
        t = self.params[:,:,GABOR_PARAM_T]
                
        pairwise_constraints = [
            s - l/32,
            l/2 - s,
            t - s,
            8*s - t
        ]
                
        # f x m x k
        self.constraints = tf.stack( box_constraints + pairwise_constraints,
                                    axis=2, name='constraints' )

        # f x m x k
        con_sqr = tf.minimum(self.constraints, 0)**2

        # f x m
        self.con_losses = tf.reduce_sum(con_sqr, axis=2, name='con_losses')

        # f (sum across mini-batch)
        self.con_loss_per_fit = tf.reduce_sum(self.con_losses, axis=1,
                                              name='con_loss_per_fit')

        ############################################################
        # Compute loss for approximation error
        
        # f x h x w
        self.err = tf.multiply((target - self.approx),
                                weight, name='err')

        err_sqr = 0.5*self.err**2

        # f (average across h/w)
        self.err_loss_per_fit = tf.reduce_mean(err_sqr, axis=(1,2),
                                               name='err_loss_per_fit')

        ############################################################
        # Compute various sums/means of above losses:

        # f
        self.loss_per_fit = tf.add(self.con_loss_per_fit,
                                   self.err_loss_per_fit,
                                   name='loss_per_fit')

        # scalars
        self.err_loss = tf.reduce_mean(self.err_loss_per_fit, name='err_loss')
        self.con_loss = tf.reduce_mean(self.con_loss_per_fit, name='con_loss')
        self.loss = self.err_loss + self.con_loss

        with tf.variable_scope('imfit_optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.opt.minimize(self.loss)

######################################################################
# Multinomial sampling.
            
def sample_multinomial(p, rshape):

    prefix_sum = np.cumsum(p)
    r = np.random.random(rshape) * prefix_sum[-1]

    idx = np.searchsorted(prefix_sum, r)
    assert(idx.shape == r.shape)
    
    return idx
    
######################################################################
# Multinomial sampling of weighted error using Boltzmann-style
# distribution

def sample_weighted_error(opts, inputs, err):
    
    err = inputs.weight_image * err

    p = opts.lambda_err*err**2
    p = np.exp(p - p.max())
    psum = p.sum()
    
    h, w = p.shape

    p = p.flatten()

    assert inputs.y.shape == (1, 1, h, 1)
    assert inputs.x.shape == (1, 1, 1, w)

    rshape = (opts.num_local, opts.mini_ensemble_size)
    idx = sample_multinomial(p, rshape)
    
    row = idx / w
    col = idx % w

    assert row.min() >= 0 and row.max() < h
    assert col.min() >= 0 and col.max() < w

    u = inputs.x.flatten()[col]
    v = inputs.y.flatten()[row]

    uv = np.stack((u, v), axis=2)

    xf = inputs.x.flatten()
    yf = inputs.y.flatten()
    px = xf[1] - xf[0]
    uv += (np.random.random(uv.shape)-0.5)*px

    '''
    uv = uv.reshape((-1, 2))
    import matplotlib.pyplot as plt
    plt.pcolormesh(xf, yf, err)
    plt.plot(uv[:,0], uv[:,1], 'k.', markersize=2)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    sys.exit(0)
    '''

    return uv

######################################################################
# Rescale image to map given bounds to [0,255] uint8

def rescale(idata, imin, imax):

    assert imax > imin
    img = (idata - imin) / (imax - imin)
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)

    return img

######################################################################
# Apply padding to right hand side of image

def pad_right(img, width):

    if img.shape[1] >= width:
        return img
    else:
        out_shape = (img.shape[0], width-img.shape[1])
        padding = np.zeros(out_shape, dtype=img.dtype)
        return np.hstack(( img, padding ))

######################################################################
# Save a snapshot of the current state to a PNG file

def snapshot(cur_gabor, cur_approx,
             opts, inputs, models, sess,
             loop_count, model_start_idx,
             full_iteration):

    if not opts.label_snapshot:
        outfile = '{}.png'.format(opts.snapshot_prefix)
    elif isinstance(full_iteration, int):
        outfile = '{}{:04d}_{:06d}.png'.format(
            opts.snapshot_prefix, loop_count+1, full_iteration+1)
    else:
        outfile = '{}{:04d}{}.png'.format(
            opts.snapshot_prefix, loop_count+1, full_iteration)
        
    cur_abserr = np.abs(cur_approx - inputs.input_image)
    cur_abserr = cur_abserr * inputs.weight_image

    out_img = np.hstack(( rescale(inputs.input_image, -1, 1),
                          rescale(cur_approx, -1, 1),
                          rescale(cur_gabor, -1, 1),
                          rescale(cur_abserr, 0, 1.0) ))

    if opts.preview_size:

        max_rowval = min(model_start_idx, opts.num_models)

        feed_dict = { inputs.max_row: max_rowval }

        pparams = sess.run(models.preview.params)
        preview_image = sess.run(models.preview.approx, feed_dict)[0]
        pgabors = sess.run(models.preview.gabor, feed_dict).sum(axis=(0,2,3))
        
        if np.any(np.isnan(preview_image)):
            print(preview_image)
            print(pparams[:max_rowval])
            print('# params:', np.isnan(pparams[:max_rowval]).sum())
            print(np.isnan(pgabors))
            assert not np.any(np.isnan(preview_image))
        assert not np.all(preview_image == 0)
        preview_image = rescale(preview_image, -1, 1)

        max_width = max(preview_image.shape[1], out_img.shape[1])
        
        preview_image = pad_right(preview_image, max_width)
        
        out_img = pad_right(out_img, max_width)
        out_img = np.vstack( (out_img, preview_image) )
    
    out_img = Image.fromarray(out_img, 'L')

    out_img.save(outfile)

######################################################################
# Apply a small perturbation to the input parameters

def randomize(params, rstdev, ncopy=None):

    gmin = GABOR_RANGE[:,0]
    gmax = GABOR_RANGE[:,1]
    grng = gmax - gmin

    pshape = params.shape
    if ncopy is not None:
        pshape = (ncopy,) + pshape

    bump = np.random.normal(scale=rstdev, size=pshape)
    
    return params + bump*grng[None,:]

######################################################################
# Compute x/y coordinates for a grid spanning [-1, 1] for the given
# image shape (h, w)

def normalized_grid(shape):

    h, w = shape
    hwmax = max(h, w)

    px = 2.0 / hwmax

    x = (np.arange(w, dtype=np.float32) - 0.5*(w) + 0.5) * px
    y = (np.arange(h, dtype=np.float32) - 0.5*(h) + 0.5) * px

    # shape broadcastable to 1 x 1 x h x w
    x = x.reshape(1, 1,  1, -1)
    y = y.reshape(1, 1, -1,  1)

    return px, x, y

######################################################################
# Set up all of the tensorflow inputs to our models

def setup_inputs(opts):

    input_image = open_grayscale(opts.image, opts.max_size)

    if opts.weights is not None:
        weight_image = open_grayscale(opts.weights, opts.max_size)
        assert weight_image.size == input_image.size
    else:
        weight_image = 1.0

    # move to -1, 1 range for input image
    input_image = input_image * 2 - 1
    
    px, x, y = normalized_grid(input_image.shape)

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
    
    target_tensor = tf.placeholder(tf.float32,
                                   shape=input_image.shape,
                                   name='target')

    max_row = tf.placeholder(tf.int32, shape=(),
                             name='max_row')

    return InputsTuple(input_image, weight_image,
                       x, y, target_tensor, max_row)

######################################################################
# Set up tensorflow models themselves. We need a separate model for
# each combination of inputs/dimensions to optimize.

def setup_models(opts, inputs):
    
    weight_tensor = tf.constant(inputs.weight_image)

    x_tensor = tf.constant(inputs.x)
    y_tensor = tf.constant(inputs.y)

    with tf.variable_scope('full'):

        full = GaborModel(x_tensor, y_tensor,
                          (1, opts.num_models),
                          weight_tensor,
                          inputs.target_tensor,
                          learning_rate=opts.full_learning_rate,
                          max_row = inputs.max_row,
                          initializer=tf.zeros_initializer())
    
    with tf.variable_scope('local'):
        
        local = GaborModel(x_tensor, y_tensor,
                           (opts.num_local, opts.mini_ensemble_size),
                           weight_tensor,
                           inputs.target_tensor,
                           learning_rate=opts.local_learning_rate)
        

    if opts.preview_size:

        preview_shape = scale_shape(map(int, inputs.target_tensor.shape),
                                    opts.preview_size)

        _, x_preview, y_preview = normalized_grid(preview_shape)
        
        with tf.variable_scope('preview'):
            preview = GaborModel(x_preview, y_preview,
                                 (1, opts.num_models),
                                 weight_tensor,
                                 target=None,
                                 max_row=inputs.max_row,
                                 params=full.params)

    else:

        preview = None

    return ModelsTuple(full, local, preview)

######################################################################
# Load weights from file.

def load_params(opts, inputs, models, state, sess):

    iparams = np.genfromtxt(opts.input, dtype=np.float32, delimiter=',')
    nparams = len(iparams)

    print('loaded {} models from {}'.format(
        nparams, opts.input))

    nparams -= nparams % opts.mini_ensemble_size
    nparams = min(nparams, opts.num_models)

    if nparams < len(iparams):
        print('warning: truncating input to {} models '
              'by randomly discarding {} models!'.format(
                  nparams, len(iparams)-nparams))
        idx = np.arange(len(iparams))
        np.random.shuffle(idx)
        iparams = iparams[idx[:nparams]]

    state.params[:nparams] = iparams

    models.full.params.load(state.params[None,:,:], sess)

    fetches = dict(gabor=models.full.gabor,
                   approx=models.full.approx,
                   err_loss=models.full.err_loss,
                   loss=models.full.loss,
                   con_losses=models.full.con_losses)

    feed_dict = {inputs.target_tensor: inputs.input_image,
                 inputs.max_row: nparams}

    results = sess.run(fetches, feed_dict)

    state.gabor[:nparams] = results['gabor'][0, :nparams]
    state.con_loss[:nparams] = results['con_losses'][0, :nparams]

    prev_best_loss = results['err_loss'] + state.con_loss[:nparams].sum()

    cur_gabor = results['approx'][0]

    if opts.preview_size:
        models.full.params.load(state.params[None,:])
    
    snapshot(cur_gabor, cur_gabor, opts, inputs, models, sess, -1, nparams, '')
    
    print('initial loss is {}'.format(prev_best_loss))
    print()

    model_start_idx = nparams

    return prev_best_loss, model_start_idx

######################################################################
# Set up state variables to record weights, Gabor approximations, &
# losses that need to persist across loops.

def setup_state(opts, inputs):
    
    state = StateTuple(
        
        params=np.zeros((opts.num_models, GABOR_NUM_PARAMS),
                        dtype=np.float32),
        
        gabor=np.zeros((opts.num_models,) + inputs.input_image.shape,
                       dtype=np.float32),

        con_loss=np.zeros(opts.num_models, dtype=np.float32)

    )

    return state

######################################################################
# Perform a deep copy of a state

def copy_state(state):
    return StateTuple(*[x.copy() for x in state])

######################################################################
# Perform an optimization on the full joint model (expensive/slow).

def full_optimize(opts, inputs, models, state, sess,
                  loop_count,
                  model_start_idx,
                  prev_best_loss,
                  rollback_loss):

    print('performing full optimization')
    print('  before full opt, loss: {}'.format(prev_best_loss))
    
    if rollback_loss is not None:
        print('  best prev full loss is {}'.format(rollback_loss))

    models.full.params.load(state.params[None,:], sess)

    max_rowval = min(model_start_idx, opts.num_models)

    feed_dict = { inputs.target_tensor: inputs.input_image,
                  inputs.max_row: max_rowval }

    fetches = dict(gabor=models.full.gabor,
                   approx=models.full.approx,
                   params=models.full.cparams,
                   loss=models.full.loss,
                   train_op=models.full.train_op,
                   con_losses=models.full.con_losses)

    for i in range(opts.full_iter):

        sess.run(models.full.train_op, feed_dict)

        if ((i+1) % 1000 == 0 and (i+1) < opts.full_iter):

            results = sess.run(fetches, feed_dict)

            print('  loss at iter {:6d} is {}'.format(
                i+1, results['loss']))

            snapshot(results['approx'][0],
                     results['approx'][0],
                     opts, inputs, models, sess,
                     loop_count, model_start_idx, i)

    results = sess.run(fetches, feed_dict)
                         
    print('  new final loss is now  {}'.format(results['loss']))

    snapshot(results['approx'][0],
             results['approx'][0],
             opts, inputs, models, sess,
             loop_count, model_start_idx, opts.full_iter-1)

    if results['loss'] < prev_best_loss:

        state.params[:max_rowval] = results['params'][0]
        state.gabor[:max_rowval] = results['gabor'][0]
        state.con_loss[:max_rowval] = results['con_losses'][0]
        prev_best_loss = results['loss']

    print()

    return prev_best_loss

######################################################################
# Optimize a bunch of randomly-initialized small ensembles in
# parallel.

def local_optimize(opts, inputs, models, state, sess,
                   cur_approx, cur_con_losses, cur_target,
                   is_replace, model_idx, loop_count,
                   model_start_idx, prev_best_loss):

    if prev_best_loss is not None:
        print('  loss before local fit is', prev_best_loss)
        
    # Params have already been randomly initialized, but we
    # need to replace some of them here

    if is_replace or opts.lambda_err:

        # Get current randomly initialized values
        pvalues = sess.run(models.local.params)

        if opts.lambda_err:
            # Do Boltzmann-style sampling of error for u,v
            pvalues[:, :, :2] = sample_weighted_error(opts, inputs,
                                                           cur_target)


        if is_replace and opts.copy_quantity:

            # Load in existing model values, slightly perturbed.
            rparams = randomize(state.params[model_idx],
                                opts.perturb_amount,
                                opts.copy_quantity)
            
            pvalues[:opts.copy_quantity] = rparams
            
        # Update tensor with data set above
        models.local.params.load(pvalues, sess)

    feed_dict = {inputs.target_tensor: cur_target}

    fetches = dict(loss=models.local.loss_per_fit,
                   con_losses=models.local.con_losses,
                   approx=models.local.approx,
                   gabor=models.local.gabor,
                   params=models.local.cparams)
        
    for i in range(opts.local_iter):
        results = sess.run(models.local.train_op, feed_dict)
                         
    results = sess.run(fetches, feed_dict)

    fidx = results['loss'].argmin()

    new_loss = results['loss'][fidx] + cur_con_losses
    new_approx = results['approx'][fidx]
    new_gabor = results['gabor'][fidx]
    new_params = results['params'][fidx]
    new_con_loss = results['con_losses'][fidx]
    
    print('  after local fit, loss is', new_loss)

    assert(new_params.shape == (opts.mini_ensemble_size, 8))

    assert(new_con_loss.shape == (opts.mini_ensemble_size,))

    assert(new_gabor.shape ==
           (opts.mini_ensemble_size,) + inputs.input_image.shape)

    assert(new_approx.shape == inputs.input_image.shape)
    
    if opts.preview_size:
        tmpparams = state.params.copy()
        tmpparams[model_idx] = new_params
        models.full.params.load(tmpparams[None,:], sess)

    snapshot(new_approx,
             cur_approx + new_approx,
             opts, inputs, models, sess,
             loop_count, model_start_idx+opts.mini_ensemble_size, '')

    if prev_best_loss is None or new_loss < prev_best_loss:
        
        do_update = True
        
    else:

        rel_change = (prev_best_loss - new_loss) / prev_best_loss

        if not opts.anneal_temp:
            print('  not better than', prev_best_loss, 'skipping update')
            do_update = False
        else:
            p_accept = np.exp(rel_change / opts.anneal_temp)
            r = np.random.random()
            do_update = (r < p_accept)
            action = 'accepting' if do_update else 'rejecting'
            print('  {} relative increase of {}, p={}'.format(
                action, -rel_change, p_accept))
    
    if do_update:

        prev_best_loss = new_loss

        state.params[model_idx] = new_params
        state.gabor[model_idx] = new_gabor
        state.con_loss[model_idx] = new_con_loss


    print()

    return prev_best_loss

######################################################################
# Our main function

def main():

    ############################################################
    # Set up variables
    
    opts = get_options()

    inputs = setup_inputs(opts)
    models = setup_models(opts, inputs)
    state = setup_state(opts, inputs)

    prev_best_loss = None
    model_start_idx = 0

    rollback_state = None
    rollback_loss = None

    ############################################################
    # Finalize the graph before doing anything with a tf.Session() -
    # this can help catch performance killers caused by adding to the
    # tensorflow graph in a loop (which ends up costing O(n^2) over
    # the entire loop).

    ginit = tf.global_variables_initializer()

    tf.get_default_graph().finalize()

    # Tell tf not to blather on console
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ############################################################
    # Now start a tensorflow session and get to work!
    
    with tf.Session() as sess:

        # Initialize all global vars (including optimizer-internal vars)
        sess.run(ginit)

        # Get start time
        start_time = datetime.now()
        
        # Parse input file
        if opts.input is not None:
            
            prev_best_loss, model_start_idx = load_params(opts, inputs,
                                                          models, state,
                                                          sess)

            loop_count = -1

            if opts.time_limit != 0 and opts.total_iterations != 0:
                
                prev_best_loss = full_optimize(opts, inputs, models, state,
                                               sess,
                                               loop_count,
                                               model_start_idx,
                                               prev_best_loss,
                                               rollback_loss)

                if opts.output is not None:
                    np.savetxt(opts.output, state.params,
                               fmt='%f', delimiter=',')

            rollback_state = copy_state(state)
            rollback_loss = prev_best_loss
                    
        loop_count = 0
                    
        # Optimization loop (hit Ctrl+C to quit)
        while True:

            if opts.time_limit is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > opts.time_limit:
                    print('exceeded time limit of {}s, quitting!'.format(
                        opts.time_limit))
                    break

            if ( opts.total_iterations is not None and
                 loop_count >= opts.total_iterations ):
                print('reached {} outer loop iterations, quitting!'.format(
                    opts.total_iterations))
                break

            # Initialize all global vars (including optimizer-internal vars)
            sess.run(ginit)

            # Establish starting index for models and whether to replace or not
            is_replace = (model_start_idx >= opts.num_models)

            print('at loop iteration {}, '.format(loop_count+1), end='')
            
            # Figure out which model(s) to replace or newly train
            if is_replace:
                
                idx = np.arange(opts.num_models)
                np.random.shuffle(idx)

                model_idx = idx[-opts.mini_ensemble_size:]
                rest_idx = idx[:-opts.mini_ensemble_size]
                
                print('replacing models', model_idx)
                
            else:
                
                model_idx = np.arange(opts.mini_ensemble_size) + model_start_idx
                rest_idx = np.arange(model_start_idx)
                
                print('training models', model_idx)
                
            # Get the current approximation (sum of all Gabor functions
            # from all models except the current ones)
            cur_approx = state.gabor[rest_idx].sum(axis=0)
 
            # The function to fit is the difference betw. input image
            # and current approximation so far.
            cur_target = inputs.input_image - cur_approx
           
            # Have to track constraint losses separately from
            # approximation error losses
            cur_con_losses = state.con_loss[rest_idx].sum()

            # Do a big parallel optimization for a bunch of random
            # model initializations
            prev_best_loss = local_optimize(opts, inputs, models,
                                            state, sess,
                                            cur_approx, cur_con_losses,
                                            cur_target,
                                            is_replace, model_idx, 
                                            loop_count,
                                            model_start_idx,
                                            prev_best_loss)

            # Done with this mini-ensemble
            model_start_idx += opts.mini_ensemble_size

            if ( model_start_idx >= opts.num_models and
                 (loop_count + 1) % opts.full_every == 0 ):

                # Do a full optimization
                prev_best_loss = full_optimize(opts, inputs, models, state,
                                               sess,
                                               loop_count,
                                               model_start_idx,
                                               prev_best_loss,
                                               rollback_loss)
                
                if rollback_loss is None or prev_best_loss <= rollback_loss:
                    rollback_loss = prev_best_loss
                    rollback_state = copy_state(state)
                    print('current loss of {} is best so far!\n'.format(
                        rollback_loss))
                else:
                    print('cur. loss of {} is not better than prev. {}, rolling back!!!\n'.format(
                        prev_best_loss, rollback_loss))
                    prev_best_loss = rollback_loss
                    state = copy_state(rollback_state)
                    
                if opts.output is not None:
                    np.savetxt(opts.output, state.params,
                               fmt='%f', delimiter=',')
                
            # Finished with this loop iteration
            loop_count += 1
            
if __name__ == '__main__':
    main()

