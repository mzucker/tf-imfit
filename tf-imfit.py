from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image

COLORMAP = None

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
                        help='size of large preview image (0 to disable)')
    
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
                        default=0.0005)

    parser.add_argument('-P', '--perturb-amount', type=float,
                        metavar='R', default=0.15,
                        help='amount to perturb replacement fits by')

    parser.add_argument('-c', '--copy-quantity', type=float,
                        metavar='C',
                        help='number or fraction of re-fits to initialize with cur. model',
                        default=0.5)
    
    parser.add_argument('-a', '--anneal-temp', type=float, metavar='T',
                        help='temperature for simulated annealing',
                        default=0.08)
        
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

    if opts.preview_size < 0:
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
        l = self.cparams[:,:,GABOR_PARAM_L]
        s = self.cparams[:,:,GABOR_PARAM_S]
        t = self.cparams[:,:,GABOR_PARAM_T]
                
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
# Rescale image to map given bounds to [0,255] uint8

def rescale(idata, imin, imax, colormap=None):

    assert imax > imin
    img = (idata - imin) / (imax - imin)
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)

    if colormap is None:
        img = np.dstack( (img, img, img) )
    else:
        img = colormap[img]

    return img

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

    if cur_gabor is None:
        cur_gabor = np.zeros_like(cur_approx)
        
    cur_abserr = np.abs(cur_approx - inputs.input_image)
    cur_abserr = cur_abserr * inputs.weight_image
    #cur_abserr = 1 - np.exp( -5*cur_abserr )
    cur_abserr = np.power(cur_abserr, 0.5)

    global COLORMAP
    
    if COLORMAP is None:
        COLORMAP = get_colormap()

    if not opts.preview_size:
        
        out_img = np.hstack(( rescale(inputs.input_image, -1, 1),
                              rescale(cur_approx, -1, 1),
                              rescale(cur_gabor, -1, 1),
                              rescale(cur_abserr, 0, 1.0, COLORMAP) ))

    else:
        
        max_rowval = min(model_start_idx, opts.num_models)

        feed_dict = { inputs.max_row: max_rowval }

        preview_image = sess.run(models.preview.approx, feed_dict)[0]
        ph, pw = preview_image.shape
        preview_image = rescale(preview_image, -1, 1)

        err_image = rescale(cur_abserr, 0, 1.0, COLORMAP)
        err_image = Image.fromarray(err_image, 'RGB')

        err_image = err_image.resize( (pw, ph),
                                      resample=Image.NEAREST )
        
        err_image = np.array(err_image)

        out_img = np.hstack((preview_image, err_image))
    
    out_img = Image.fromarray(out_img, 'RGB')

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

    if opts.input is not None:
        iparams = np.genfromtxt(opts.input, dtype=np.float32, delimiter=',')
        nparams = len(iparams)
    else:
        iparams = np.empty((0, GABOR_NUM_PARAMS), dtype=np.float32)
        nparams = 0 

    print('loaded {} models from {}'.format(
        nparams, opts.input))

    for i, pname in enumerate('uvrpltsh'):
        p = iparams[:,i]
        print('param {} has min {} and max {}'.format(
            pname, p.min(), p.max()))
        

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

    cur_approx = results['approx'][0]

    print('cur_approx has min {} max {} mean {} std {}'.format(
        cur_approx.min(), cur_approx.max(), cur_approx.mean(), cur_approx.std()))

    if opts.preview_size:
        models.full.params.load(state.params[None,:])
    
    snapshot(None, cur_approx,
             opts, inputs, models, sess, -1, nparams, '')
    
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

    if rollback_loss is not None:
        print('  best prev full loss is {}'.format(rollback_loss))
    
    print('  before full opt, loss: {}'.format(prev_best_loss))

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

            snapshot(None,
                     results['approx'][0],
                     opts, inputs, models, sess,
                     loop_count, model_start_idx, i)

    results = sess.run(fetches, feed_dict)
                         
    print('  new final loss is now  {}'.format(results['loss']))

    snapshot(None,
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

    if is_replace and opts.copy_quantity:

        # Get current randomly initialized values
        pvalues = sess.run(models.local.params)

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
        prev_best_loss, model_start_idx = load_params(opts, inputs,
                                                      models, state,
                                                      sess)

        if opts.input is not None:

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
            
######################################################################

# from https://github.com/BIDS/colormap/blob/master/colormaps.py
# licensed
_magma_data = [[0.001462, 0.000466, 0.013866],
               [0.002258, 0.001295, 0.018331],
               [0.003279, 0.002305, 0.023708],
               [0.004512, 0.003490, 0.029965],
               [0.005950, 0.004843, 0.037130],
               [0.007588, 0.006356, 0.044973],
               [0.009426, 0.008022, 0.052844],
               [0.011465, 0.009828, 0.060750],
               [0.013708, 0.011771, 0.068667],
               [0.016156, 0.013840, 0.076603],
               [0.018815, 0.016026, 0.084584],
               [0.021692, 0.018320, 0.092610],
               [0.024792, 0.020715, 0.100676],
               [0.028123, 0.023201, 0.108787],
               [0.031696, 0.025765, 0.116965],
               [0.035520, 0.028397, 0.125209],
               [0.039608, 0.031090, 0.133515],
               [0.043830, 0.033830, 0.141886],
               [0.048062, 0.036607, 0.150327],
               [0.052320, 0.039407, 0.158841],
               [0.056615, 0.042160, 0.167446],
               [0.060949, 0.044794, 0.176129],
               [0.065330, 0.047318, 0.184892],
               [0.069764, 0.049726, 0.193735],
               [0.074257, 0.052017, 0.202660],
               [0.078815, 0.054184, 0.211667],
               [0.083446, 0.056225, 0.220755],
               [0.088155, 0.058133, 0.229922],
               [0.092949, 0.059904, 0.239164],
               [0.097833, 0.061531, 0.248477],
               [0.102815, 0.063010, 0.257854],
               [0.107899, 0.064335, 0.267289],
               [0.113094, 0.065492, 0.276784],
               [0.118405, 0.066479, 0.286321],
               [0.123833, 0.067295, 0.295879],
               [0.129380, 0.067935, 0.305443],
               [0.135053, 0.068391, 0.315000],
               [0.140858, 0.068654, 0.324538],
               [0.146785, 0.068738, 0.334011],
               [0.152839, 0.068637, 0.343404],
               [0.159018, 0.068354, 0.352688],
               [0.165308, 0.067911, 0.361816],
               [0.171713, 0.067305, 0.370771],
               [0.178212, 0.066576, 0.379497],
               [0.184801, 0.065732, 0.387973],
               [0.191460, 0.064818, 0.396152],
               [0.198177, 0.063862, 0.404009],
               [0.204935, 0.062907, 0.411514],
               [0.211718, 0.061992, 0.418647],
               [0.218512, 0.061158, 0.425392],
               [0.225302, 0.060445, 0.431742],
               [0.232077, 0.059889, 0.437695],
               [0.238826, 0.059517, 0.443256],
               [0.245543, 0.059352, 0.448436],
               [0.252220, 0.059415, 0.453248],
               [0.258857, 0.059706, 0.457710],
               [0.265447, 0.060237, 0.461840],
               [0.271994, 0.060994, 0.465660],
               [0.278493, 0.061978, 0.469190],
               [0.284951, 0.063168, 0.472451],
               [0.291366, 0.064553, 0.475462],
               [0.297740, 0.066117, 0.478243],
               [0.304081, 0.067835, 0.480812],
               [0.310382, 0.069702, 0.483186],
               [0.316654, 0.071690, 0.485380],
               [0.322899, 0.073782, 0.487408],
               [0.329114, 0.075972, 0.489287],
               [0.335308, 0.078236, 0.491024],
               [0.341482, 0.080564, 0.492631],
               [0.347636, 0.082946, 0.494121],
               [0.353773, 0.085373, 0.495501],
               [0.359898, 0.087831, 0.496778],
               [0.366012, 0.090314, 0.497960],
               [0.372116, 0.092816, 0.499053],
               [0.378211, 0.095332, 0.500067],
               [0.384299, 0.097855, 0.501002],
               [0.390384, 0.100379, 0.501864],
               [0.396467, 0.102902, 0.502658],
               [0.402548, 0.105420, 0.503386],
               [0.408629, 0.107930, 0.504052],
               [0.414709, 0.110431, 0.504662],
               [0.420791, 0.112920, 0.505215],
               [0.426877, 0.115395, 0.505714],
               [0.432967, 0.117855, 0.506160],
               [0.439062, 0.120298, 0.506555],
               [0.445163, 0.122724, 0.506901],
               [0.451271, 0.125132, 0.507198],
               [0.457386, 0.127522, 0.507448],
               [0.463508, 0.129893, 0.507652],
               [0.469640, 0.132245, 0.507809],
               [0.475780, 0.134577, 0.507921],
               [0.481929, 0.136891, 0.507989],
               [0.488088, 0.139186, 0.508011],
               [0.494258, 0.141462, 0.507988],
               [0.500438, 0.143719, 0.507920],
               [0.506629, 0.145958, 0.507806],
               [0.512831, 0.148179, 0.507648],
               [0.519045, 0.150383, 0.507443],
               [0.525270, 0.152569, 0.507192],
               [0.531507, 0.154739, 0.506895],
               [0.537755, 0.156894, 0.506551],
               [0.544015, 0.159033, 0.506159],
               [0.550287, 0.161158, 0.505719],
               [0.556571, 0.163269, 0.505230],
               [0.562866, 0.165368, 0.504692],
               [0.569172, 0.167454, 0.504105],
               [0.575490, 0.169530, 0.503466],
               [0.581819, 0.171596, 0.502777],
               [0.588158, 0.173652, 0.502035],
               [0.594508, 0.175701, 0.501241],
               [0.600868, 0.177743, 0.500394],
               [0.607238, 0.179779, 0.499492],
               [0.613617, 0.181811, 0.498536],
               [0.620005, 0.183840, 0.497524],
               [0.626401, 0.185867, 0.496456],
               [0.632805, 0.187893, 0.495332],
               [0.639216, 0.189921, 0.494150],
               [0.645633, 0.191952, 0.492910],
               [0.652056, 0.193986, 0.491611],
               [0.658483, 0.196027, 0.490253],
               [0.664915, 0.198075, 0.488836],
               [0.671349, 0.200133, 0.487358],
               [0.677786, 0.202203, 0.485819],
               [0.684224, 0.204286, 0.484219],
               [0.690661, 0.206384, 0.482558],
               [0.697098, 0.208501, 0.480835],
               [0.703532, 0.210638, 0.479049],
               [0.709962, 0.212797, 0.477201],
               [0.716387, 0.214982, 0.475290],
               [0.722805, 0.217194, 0.473316],
               [0.729216, 0.219437, 0.471279],
               [0.735616, 0.221713, 0.469180],
               [0.742004, 0.224025, 0.467018],
               [0.748378, 0.226377, 0.464794],
               [0.754737, 0.228772, 0.462509],
               [0.761077, 0.231214, 0.460162],
               [0.767398, 0.233705, 0.457755],
               [0.773695, 0.236249, 0.455289],
               [0.779968, 0.238851, 0.452765],
               [0.786212, 0.241514, 0.450184],
               [0.792427, 0.244242, 0.447543],
               [0.798608, 0.247040, 0.444848],
               [0.804752, 0.249911, 0.442102],
               [0.810855, 0.252861, 0.439305],
               [0.816914, 0.255895, 0.436461],
               [0.822926, 0.259016, 0.433573],
               [0.828886, 0.262229, 0.430644],
               [0.834791, 0.265540, 0.427671],
               [0.840636, 0.268953, 0.424666],
               [0.846416, 0.272473, 0.421631],
               [0.852126, 0.276106, 0.418573],
               [0.857763, 0.279857, 0.415496],
               [0.863320, 0.283729, 0.412403],
               [0.868793, 0.287728, 0.409303],
               [0.874176, 0.291859, 0.406205],
               [0.879464, 0.296125, 0.403118],
               [0.884651, 0.300530, 0.400047],
               [0.889731, 0.305079, 0.397002],
               [0.894700, 0.309773, 0.393995],
               [0.899552, 0.314616, 0.391037],
               [0.904281, 0.319610, 0.388137],
               [0.908884, 0.324755, 0.385308],
               [0.913354, 0.330052, 0.382563],
               [0.917689, 0.335500, 0.379915],
               [0.921884, 0.341098, 0.377376],
               [0.925937, 0.346844, 0.374959],
               [0.929845, 0.352734, 0.372677],
               [0.933606, 0.358764, 0.370541],
               [0.937221, 0.364929, 0.368567],
               [0.940687, 0.371224, 0.366762],
               [0.944006, 0.377643, 0.365136],
               [0.947180, 0.384178, 0.363701],
               [0.950210, 0.390820, 0.362468],
               [0.953099, 0.397563, 0.361438],
               [0.955849, 0.404400, 0.360619],
               [0.958464, 0.411324, 0.360014],
               [0.960949, 0.418323, 0.359630],
               [0.963310, 0.425390, 0.359469],
               [0.965549, 0.432519, 0.359529],
               [0.967671, 0.439703, 0.359810],
               [0.969680, 0.446936, 0.360311],
               [0.971582, 0.454210, 0.361030],
               [0.973381, 0.461520, 0.361965],
               [0.975082, 0.468861, 0.363111],
               [0.976690, 0.476226, 0.364466],
               [0.978210, 0.483612, 0.366025],
               [0.979645, 0.491014, 0.367783],
               [0.981000, 0.498428, 0.369734],
               [0.982279, 0.505851, 0.371874],
               [0.983485, 0.513280, 0.374198],
               [0.984622, 0.520713, 0.376698],
               [0.985693, 0.528148, 0.379371],
               [0.986700, 0.535582, 0.382210],
               [0.987646, 0.543015, 0.385210],
               [0.988533, 0.550446, 0.388365],
               [0.989363, 0.557873, 0.391671],
               [0.990138, 0.565296, 0.395122],
               [0.990871, 0.572706, 0.398714],
               [0.991558, 0.580107, 0.402441],
               [0.992196, 0.587502, 0.406299],
               [0.992785, 0.594891, 0.410283],
               [0.993326, 0.602275, 0.414390],
               [0.993834, 0.609644, 0.418613],
               [0.994309, 0.616999, 0.422950],
               [0.994738, 0.624350, 0.427397],
               [0.995122, 0.631696, 0.431951],
               [0.995480, 0.639027, 0.436607],
               [0.995810, 0.646344, 0.441361],
               [0.996096, 0.653659, 0.446213],
               [0.996341, 0.660969, 0.451160],
               [0.996580, 0.668256, 0.456192],
               [0.996775, 0.675541, 0.461314],
               [0.996925, 0.682828, 0.466526],
               [0.997077, 0.690088, 0.471811],
               [0.997186, 0.697349, 0.477182],
               [0.997254, 0.704611, 0.482635],
               [0.997325, 0.711848, 0.488154],
               [0.997351, 0.719089, 0.493755],
               [0.997351, 0.726324, 0.499428],
               [0.997341, 0.733545, 0.505167],
               [0.997285, 0.740772, 0.510983],
               [0.997228, 0.747981, 0.516859],
               [0.997138, 0.755190, 0.522806],
               [0.997019, 0.762398, 0.528821],
               [0.996898, 0.769591, 0.534892],
               [0.996727, 0.776795, 0.541039],
               [0.996571, 0.783977, 0.547233],
               [0.996369, 0.791167, 0.553499],
               [0.996162, 0.798348, 0.559820],
               [0.995932, 0.805527, 0.566202],
               [0.995680, 0.812706, 0.572645],
               [0.995424, 0.819875, 0.579140],
               [0.995131, 0.827052, 0.585701],
               [0.994851, 0.834213, 0.592307],
               [0.994524, 0.841387, 0.598983],
               [0.994222, 0.848540, 0.605696],
               [0.993866, 0.855711, 0.612482],
               [0.993545, 0.862859, 0.619299],
               [0.993170, 0.870024, 0.626189],
               [0.992831, 0.877168, 0.633109],
               [0.992440, 0.884330, 0.640099],
               [0.992089, 0.891470, 0.647116],
               [0.991688, 0.898627, 0.654202],
               [0.991332, 0.905763, 0.661309],
               [0.990930, 0.912915, 0.668481],
               [0.990570, 0.920049, 0.675675],
               [0.990175, 0.927196, 0.682926],
               [0.989815, 0.934329, 0.690198],
               [0.989434, 0.941470, 0.697519],
               [0.989077, 0.948604, 0.704863],
               [0.988717, 0.955742, 0.712242],
               [0.988367, 0.962878, 0.719649],
               [0.988033, 0.970012, 0.727077],
               [0.987691, 0.977154, 0.734536],
               [0.987387, 0.984288, 0.742002],
               [0.987053, 0.991438, 0.749504]]    

######################################################################

def get_colormap():
    return (np.array(_magma_data)*255).astype(np.uint8)

######################################################################

if __name__ == '__main__':
    main()
