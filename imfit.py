from __future__ import print_function
import re, os, sys, argparse
from datetime import datetime
from collections import namedtuple
import tensorflow as tf
import numpy as np
from PIL import Image


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

    parser.add_argument('-t', '--time-limit', type=parse_duration,
                        metavar='LIMIT',
                        help='time limit (e.g. 1:30 or 1h30m)',
                        default=None)

    parser.add_argument('-n', '--num-models', type=int, metavar='N',
                        help='number of models to fit',
                        default=128)

    parser.add_argument('-p', '--num-parallel', type=int, metavar='N',
                        help='number of random guesses per model',
                        default=200)

    parser.add_argument('-m', '--max-iter', type=int, metavar='N',
                        help='maximum # of iterations per trial fit',
                        default=100)

    parser.add_argument('-r', '--refine', type=int, metavar='N',
                        help='maximum # of iterations for final fit',
                        default=500)

    parser.add_argument('-R', '--rstdev', type=float, metavar='R',
                        help='amount to randomize models when re-fitting',
                        default=0.001)

    parser.add_argument('-l', '--learning-rate', type=float, metavar='R',
                        help='learning rate for AdamOptimizer',
                        default=0.001)

    parser.add_argument('-s', '--max-size', type=int, metavar='N',
                        help='maximum size of image to load',
                        default=128)

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

    parser.add_argument('-L', '--lambda-err', type=float,
                        metavar='LAMBDA',
                        help='weight on multinomial sampling of error map',
                        default=2.0)
    
    parser.add_argument('-F', '--full-every', type=int, metavar='N',
                        help='perform joint optimization after every N models')

    parser.add_argument('-f', '--full-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=10000)

    parser.add_argument('-S', '--label-snapshot', action='store_true',
                        help='individually labeled snapshot images')

    parser.add_argument('-e', '--mini-ensemble-size', type=int, metavar='N',
                        help='models to update per iteration',
                        default=1)

    parser.add_argument('-P', '--preview-size', type=int, metavar='N',
                        default=0,
                        help='size of preview image (<0 to disable)')
    
    opts = parser.parse_args()

    if opts.full_every is None:
        opts.full_every = opts.num_models

    if opts.preview_size == 0:
        opts.preview_size = 4*opts.max_size
    elif opts.preview_size < 0:
        opts.preview_size = 0
        
    assert opts.num_models % opts.mini_ensemble_size == 0
    assert opts.full_every % opts.mini_ensemble_size == 0 
       
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

        # Parameter tensor could be passed in or created here
        if params is not None:

            self.params = params

        else:

            if initializer is None:
                gmin = GABOR_RANGE[:,0].reshape(1,1,GABOR_NUM_PARAMS)
                gmax = GABOR_RANGE[:,1].reshape(1,1,GABOR_NUM_PARAMS)
                initializer = tf.random_uniform_initializer(minval=gmin,
                                                            maxval=gmax,
                                                            dtype=tf.float32)

            # f x m x 8
            self.params = tf.get_variable(
                'params',
                shape=(num_parallel, ensemble_size, GABOR_NUM_PARAMS),
                dtype=tf.float32,
                initializer=initializer)

        ############################################################
        # Now compute the Gabor function for each fit/model
        
        # f x m x 8 x 1 x 1
        params_bcast = self.params[:,:max_row,:,None,None]

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
            var = self.params[:,:,i]
            if lo != -np.pi:
                box_constraints.append( var - lo )
            if hi != np.pi:
                box_constraints.append( hi - var)

        
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
        self.con_loss_all = tf.reduce_sum(con_sqr, axis=2,
                                          name='con_loss_all')

        # f
        self.con_loss_per_fit = tf.reduce_sum(self.con_loss_all, axis=1,
                                            name='con_loss_per_fit')

        # m
        self.con_loss_per_batch = tf.reduce_mean(self.con_loss_all, axis=0,
                                               name='con_loss_per_batch')

        ############################################################
        # Compute loss for approximation error
        
        # f x h x w
        self.err = tf.multiply((target - self.approx),
                                weight, name='err')

        err_sqr = 0.5*self.err**2

        # f
        self.err_loss_per_fit = tf.reduce_mean(err_sqr, axis=(1,2),
                                               name='err_loss_per_fit')

        ############################################################
        # Compute various sums/means of above losses:

        # f
        self.loss_per_fit = tf.add(self.con_loss_per_fit,
                                   self.err_loss_per_fit,
                                   name='loss_per_fit')

        lpf_argmin = tf.argmin(self.loss_per_fit)

        # scalar
        self.losses_min = self.loss_per_fit[lpf_argmin]

        # m x 8
        self.params_min = self.params[lpf_argmin]

        # scalars
        self.err_loss = tf.reduce_mean(self.err_loss_per_fit, name='err_loss')
        self.con_loss = tf.reduce_mean(self.con_loss_per_fit, name='con_loss')
        self.loss = self.err_loss + self.con_loss

        with tf.variable_scope('imfit_optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.opt.minimize(self.loss)

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

    prefix_sum = np.cumsum(p)
    rshape = (opts.num_parallel, opts.mini_ensemble_size)
    r = np.random.random(rshape) * prefix_sum[-1]

    idx = np.searchsorted(prefix_sum, r)
    assert(idx.shape == r.shape)
    
    row = idx / w
    col = idx % w

    assert row.min() >= 0 and row.max() < h
    assert col.min() >= 0 and col.max() < w

    u = inputs.x.flatten()[col]
    v = inputs.y.flatten()[row]

    uv = np.stack((u, v), axis=2)

    xf = inputs.x.flatten()
    px = xf[1] - xf[0]
    uv += (np.random.random(uv.shape)-0.5)*px

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
             iteration, full_iteration):

    if not opts.label_snapshot:
        outfile = 'out.png'
    elif full_iteration is None:
        outfile = 'out{:04d}_single.png'.format(iteration+1)
    else:
        outfile = 'out{:04d}_{:06d}.png'.format(
            iteration+1, full_iteration+1)

    cur_abserr = np.abs(cur_approx - inputs.input_image)
    cur_abserr = cur_abserr * inputs.weight_image

    out_img = np.hstack(( rescale(inputs.input_image, -1, 1),
                          rescale(cur_approx, -1, 1),
                          rescale(cur_gabor, -1, 1),
                          rescale(cur_abserr, 0, 1.0) ))

    if opts.preview_size:

        max_rowval = min(iteration, opts.num_models)

        fetches = models.preview.approx
        feed_dict = { inputs.max_row: max_rowval }

        preview_image = sess.run(fetches, feed_dict)[0]
        preview_image = rescale(preview_image, -1, 1)

        max_width = max(preview_image.shape[1], out_img.shape[1])
        
        preview_image = pad_right(preview_image, max_width)
        
        out_img = pad_right(out_img, max_width)
        out_img = np.vstack( (out_img, preview_image) )
    
    out_img = Image.fromarray(out_img, 'L')

    out_img.save(outfile)

######################################################################
# Apply a small perturbation to the input parameters

def randomize(params, rstdev):

    gmin = GABOR_RANGE[:,0]
    gmax = GABOR_RANGE[:,1]
    grng = gmax - gmin

    bump = np.random.normal(scale=rstdev, size=params.shape)
    
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

    InputsTuple = namedtuple('InputsTuple',
                             'input_image, weight_image, '
                             'x, y, target_tensor, max_row')

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

    ModelsTuple = namedtuple('ModelsTuple',
                             'full, par_mini, one_mini, preview')
    
    weight_tensor = tf.constant(inputs.weight_image)

    x_tensor = tf.constant(inputs.x)
    y_tensor = tf.constant(inputs.y)

    with tf.variable_scope('full'):

        full = GaborModel(x_tensor, y_tensor,
                          (1, opts.num_models),
                          weight_tensor,
                          inputs.target_tensor,
                          learning_rate=opts.learning_rate,
                          max_row = inputs.max_row,
                          initializer=tf.zeros_initializer())
    
    with tf.variable_scope('par_mini'):
        
        par_mini = GaborModel(x_tensor, y_tensor,
                              (opts.num_parallel, opts.mini_ensemble_size),
                              weight_tensor,
                              inputs.target_tensor,
                              learning_rate=opts.learning_rate)
        
    with tf.variable_scope('one_mini'):
        
        one_mini = GaborModel(x_tensor, y_tensor,
                              (1, opts.mini_ensemble_size),
                              weight_tensor,
                              inputs.target_tensor,
                              learning_rate=opts.learning_rate)
        

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

    return ModelsTuple(full, par_mini, one_mini, preview)

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
                   con_losses=models.full.con_loss_per_batch)

    feed_dict = {inputs.target_tensor: inputs.input_image,
                 inputs.max_row: nparams}

    results = sess.run(fetches, feed_dict)

    state.approx[:nparams] = results['gabor'][0, :nparams]
    state.con_loss[:nparams] = results['con_losses'][:nparams]

    prev_best_loss = results['err_loss'] + state.con_loss[:nparams].sum()

    cur_gabor = results['approx'][0]

    if opts.preview_size:
        models.full.params.load(state.params[None,:])
    
    snapshot(cur_gabor, cur_gabor, opts, inputs, models, sess, nparams, 0)
    
    print('current loss is {}'.format(prev_best_loss))

    iteration = nparams

    return prev_best_loss, iteration

######################################################################
# Set up state variables to record weights, Gabor approximations, &
# losses that need to persist across loops.

def setup_state(opts, inputs):

    StateTuple = namedtuple('StateTuple', 'params, approx, con_loss')
    
    state = StateTuple(
        
        params=np.zeros((opts.num_models, GABOR_NUM_PARAMS),
                        dtype=np.float32),
        
        approx=np.zeros((opts.num_models,) + inputs.input_image.shape,
                        dtype=np.float32),
        
        con_loss=np.zeros(opts.num_models, dtype=np.float32)

    )

    return state

######################################################################
# Perform an optimization on the full joint model (expensive/slow).

def full_optimize(opts, inputs, models, state, sess,
                  prev_best_loss, iteration):

    print('performing full optimization!')
    print('  previous best loss was {}'.format(prev_best_loss))

    rparams = randomize(state.params, opts.rstdev)
    models.full.params.load(rparams[None,:,:], sess)

    feed_dict = {inputs.target_tensor: inputs.input_image,
                 inputs.max_row: opts.num_models}

    fetches = dict(gabor=models.full.gabor,
                   approx=models.full.approx,
                   params=models.full.params,
                   loss=models.full.loss,
                   train_op=models.full.train_op,
                   con_losses=models.full.con_loss_per_batch)

    for i in range(opts.full_iter):

        results = sess.run(fetches, feed_dict)

        if ((i+1) % 1000 == 0 and (i+1) < opts.full_iter):

            print('  loss at iter {:6d} is {}'.format(
                i+1, results['loss']))

            snapshot(results['approx'][0],
                     results['approx'][0],
                     opts, inputs, models, sess,
                     iteration, i)

    print('  new final loss is now  {}'.format(results['loss']))

    snapshot(results['approx'][0],
             results['approx'][0],
             opts, inputs, models, sess,
             iteration, opts.full_iter)

    if results['loss'] < prev_best_loss:
        state.params[:] = results['params'][0]
        state.approx[:] = results['gabor'][0]
        state.con_loss[:] = results['con_losses']
        prev_best_loss = results['loss']

        if opts.output is not None:
            np.savetxt(opts.output,
                       state.params[:min(iteration, opts.num_models)],
                       fmt='%f', delimiter=',')

    print()

    return prev_best_loss

######################################################################
# Optimize a bunch of randomly-initialized small ensembles in
# parallel.

def par_mini_optimize(opts, inputs, models, state, sess,
                      cur_approx, cur_con_losses, cur_target,
                      is_replace, model_idx):

    # Params have already been randomly initialized, but we
    # need to replace some of them here
    set_pvalues = is_replace or opts.lambda_err

    if set_pvalues:

        # Get current randomly initialized values
        pvalues = sess.run(models.par_mini.params)

        if opts.lambda_err:
            # Do Boltzmann-style sampling of error for u,v
            pvalues[:, :, :2] = sample_weighted_error(opts, inputs,
                                                      cur_target)

        if is_replace:
            # Load in existing model values, slightly perturbed.
            pvalues[0, :, :] = randomize(state.params[model_idx],
                                         opts.rstdev)

        # Update tensor with data set above
        models.par_mini.params.load(pvalues, sess)

    fetches = dict(params=models.par_mini.params,
                   train_op=models.par_mini.train_op,
                   best_loss=models.par_mini.losses_min,
                   best_params=models.par_mini.params_min)

    feed_dict = {inputs.target_tensor: cur_target}

    for i in range(opts.max_iter):
        results = sess.run(fetches, feed_dict)

    best_loss = results['best_loss'] + cur_con_losses
    best_params = results['best_params']

    print('  best loss so far is', best_loss)

    return best_loss, best_params

######################################################################
# Take the best-performing result from the parallel optimization
# above and refine it.

def one_mini_optimize(opts, inputs, models, state, sess,
                      cur_approx, cur_con_losses, cur_target,
                      is_replace, model_idx, best_params,
                      iteration, prev_best_loss):
    
    models.one_mini.params.load(best_params[None,:], sess)

    fetches = dict(params=models.one_mini.params,
                   con_loss_per_batch=models.one_mini.con_loss_per_batch,
                   loss=models.one_mini.loss,
                   train_op=models.one_mini.train_op,
                   gabor=models.one_mini.gabor)

    feed_dict = {inputs.target_tensor: cur_target}

    for i in range(opts.refine):
        results = sess.run(fetches, feed_dict)

    new_loss = cur_con_losses + results['loss']
    print('  post-refine loss is', new_loss)

    assert(results['params'].shape ==
           (1, opts.mini_ensemble_size, 8))

    assert(results['gabor'].shape ==
           (1, opts.mini_ensemble_size) + inputs.input_image.shape)
    
    assert(results['con_loss_per_batch'].shape ==
           (opts.mini_ensemble_size,))

    new_params = results['params'][0]
    new_approx = results['gabor'][0]
    new_con_loss = results['con_loss_per_batch']

    if (is_replace and
        prev_best_loss is not None and
        new_loss >= prev_best_loss):

        print('  not better than', prev_best_loss, 'skipping update')

    else:

        if opts.output is not None:
            np.savetxt(opts.output,
                       state.params[:min(iteration, opts.num_models)],
                       fmt='%f', delimiter=',')

        prev_best_loss = new_loss

        state.params[model_idx] = new_params
        state.approx[model_idx] = new_approx
        state.con_loss[model_idx] = new_con_loss

        cur_approx += state.approx[model_idx].sum(axis=0)

        outfile = 'out{:04d}.png'.format(iteration+1)

        if opts.preview_size:
            models.full.params.load(state.params[None,:,:], sess)

        snapshot(results['gabor'][0].sum(axis=0),
                 cur_approx,
                 opts, inputs, models, sess,
                 iteration, None)

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
    iteration = 0

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

        # Parse input file
        if opts.input is not None:
            
            if not os.path.isfile(opts.input):
                print("warning: can't load weights from", opts.input)
            else:
                prev_best_loss, iteration = load_params(opts, inputs,
                                                        models, state,
                                                        sess)

        # Get start time
        start_time = datetime.now()

        # Optimization loop (hit Ctrl+C to quit)
        while True:

            if opts.time_limit is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > opts.time_limit:
                    print('exceeded time limit of {}s, quitting!'.format(
                        opts.time_limit))
                    break

            # Initialize all global vars (including optimizer-internal vars)
            sess.run(ginit)

            # Establish starting index for models and whether to replace or not
            model_start_idx = iteration * opts.mini_ensemble_size
            is_replace = (model_start_idx >= opts.num_models)

            # See if it's time to  do a full optimization
            if (is_replace and model_start_idx % opts.full_every == 0):
                prev_best_loss = full_optimize(opts, inputs, models, state,
                                               sess, prev_best_loss, iteration)


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
            cur_approx = state.approx[rest_idx].sum(axis=0)
 
            # The function to fit is the difference betw. input image
            # and current approximation so far.
            cur_target = inputs.input_image - cur_approx
           
            # Have to track constraint losses separately from
            # approximation error losses
            cur_con_losses = state.con_loss[rest_idx].sum()

            # Do a big parallel optimization for a bunch of random
            # model initializations
            best_loss, best_params = par_mini_optimize(opts, inputs, models,
                                                       state, sess,
                                                       cur_approx, cur_con_losses,
                                                       cur_target, is_replace,
                                                       model_idx)

            # Take the best result of the mini-optimization above and
            # refine it.
            prev_best_loss = one_mini_optimize(opts, inputs, models,
                                               state, sess,
                                               cur_approx, cur_con_losses,
                                               cur_target, is_replace,
                                               model_idx, best_params,
                                               iteration, prev_best_loss)

            # Done this loop
            iteration += 1
            
if __name__ == '__main__':
    main()

