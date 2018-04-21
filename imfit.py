from __future__ import print_function
import os
import sys
import argparse
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
# Parse command-line options, return namespace containing results

def get_options():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('image', type=argparse.FileType('r'),
                        metavar='IMAGE.png',
                        help='image to approximate')

    parser.add_argument('-n', '--num-models', type=int, metavar='N',
                        help='number of models to fit',
                        default=64)

    parser.add_argument('-p', '--num-parallel', type=int, metavar='N',
                        help='number of random guesses per model',
                        default=200)

    parser.add_argument('-m', '--max-iter', type=int, metavar='N',
                        help='maximum # of iterations per trial fit',
                        default=100)

    parser.add_argument('-r', '--refine', type=int, metavar='N',
                        help='maximum # of iterations for final fit',
                        default=100)

    parser.add_argument('-R', '--rstdev', type=float, metavar='R',
                        help='amount to randomize models when re-fitting',
                        default=0.001)

    parser.add_argument('-l', '--learning-rate', type=float, metavar='R',
                        help='learning rate for AdamOptimizer',
                        default=0.001)

    parser.add_argument('-s', '--max-size', type=int, metavar='N',
                        help='maximum size of image to load',
                        default=32)

    parser.add_argument('-w', '--weights', type=argparse.FileType('r'),
                        metavar='WEIGHTS.png',
                        help='load weights from file',
                        default=512)

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
        self.gabor_sum = tf.reduce_sum(self.gabor, axis=1, name='gabor_sum')

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
        self.err = tf.multiply((target - self.gabor_sum),
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

def sample_weighted_error(x, y, err, weight, lambda_err,
                 minishape):

    if weight is not None:
        err = weight * err

    p = lambda_err*err**2
    p = np.exp(p - p.max())
    psum = p.sum()
    
    h, w = p.shape

    p = p.flatten()
        
    assert y.shape == (1, 1, h, 1)
    assert x.shape == (1, 1, 1, w)

    prefix_sum = np.cumsum(p)
    r = np.random.random(minishape) * prefix_sum[-1]

    idx = np.searchsorted(prefix_sum, r)
    assert(idx.shape == r.shape)
    
    row = idx / w
    col = idx % w

    assert row.min() >= 0 and row.max() < h
    assert col.min() >= 0 and col.max() < w

    u = x.flatten()[col]
    v = y.flatten()[row]

    uv = np.stack((u, v), axis=2)

    xf = x.flatten()
    px = xf[1] - xf[0]
    uv += (np.random.random(uv.shape)-0.5)*px

    return uv

######################################################################

def rescale(idata, imin, imax):

    assert imax > imin
    img = (idata - imin) / (imax - imin)
    img = np.clip(img, 0, 1)
    img = (img*255).astype(np.uint8)

    return img

######################################################################

def pad_right(img, width):

    if img.shape[1] >= width:
        return img
    else:
        out_shape = (img.shape[0], width-img.shape[1])
        padding = np.zeros(out_shape, dtype=img.dtype)
        return np.hstack(( img, padding ))

######################################################################

def snapshot(cur_gabor,
             cur_approx,
             input_image, 
             weight_image,
             iteration, full_iteration,
             label_snapshot,
             preview_stuff):

    if not label_snapshot:
        outfile = 'out.png'
    elif full_iteration is None:
        outfile = 'out{:04d}_single.png'.format(iteration+1)
    else:
        outfile = 'out{:04d}_{:06d}.png'.format(
            iteration+1, full_iteration+1)

    cur_abserr = np.abs(cur_approx - input_image)
    cur_abserr = cur_abserr * weight_image

    out_img = np.hstack(( rescale(input_image, -1, 1),
                          rescale(cur_approx, -1, 1),
                          rescale(cur_gabor, -1, 1),
                          rescale(cur_abserr, 0, 1.0) ))

    if preview_stuff is not None:

        sess, g_preview, max_row = preview_stuff

        max_rowval = min(iteration, g_preview.params.shape[1])

        fetches = g_preview.gabor_sum
        feed_dict = { max_row: max_rowval }

        preview_image = sess.run(fetches, feed_dict)[0]
        preview_image = rescale(preview_image, -1, 1)

        max_width = max(preview_image.shape[1], out_img.shape[1])
        
        preview_image = pad_right(preview_image, max_width)
        
        out_img = pad_right(out_img, max_width)
        out_img = np.vstack( (out_img, preview_image) )
    
    out_img = Image.fromarray(out_img, 'L')

    out_img.save(outfile)

######################################################################

def randomize(params, rstdev):

    gmin = GABOR_RANGE[:,0]
    gmax = GABOR_RANGE[:,1]
    grng = gmax - gmin

    bump = np.random.normal(scale=rstdev, size=params.shape)
    
    return params + bump*grng[None,:]    

######################################################################

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

def setup_models(opts, x, y, weight_tensor,
                 target_tensor, max_row):

    GaborModels = namedtuple('GaborModels', 'full, par_mini, one_mini, preview')

    with tf.variable_scope('full'):

        full = GaborModel(x, y,
                          (1, opts.num_models),
                          weight_tensor,
                          target_tensor,
                          learning_rate=opts.learning_rate,
                          max_row = max_row,
                          initializer=tf.zeros_initializer())
    
    with tf.variable_scope('par_mini'):
        
        par_mini = GaborModel(x, y,
                              (opts.num_parallel, opts.mini_ensemble_size),
                              weight_tensor,
                              target_tensor,
                              learning_rate=opts.learning_rate)
        
    with tf.variable_scope('one_mini'):
        
        one_mini = GaborModel(x, y,
                              (1, opts.mini_ensemble_size),
                              weight_tensor,
                              target_tensor,
                              learning_rate=opts.learning_rate)
        

    if opts.preview_size:

        preview_shape = scale_shape(map(int, target_tensor.shape),
                                    opts.preview_size)

        _, x_preview, y_preview = normalized_grid(preview_shape)
        
        with tf.variable_scope('preview'):
            preview = GaborModel(x_preview, y_preview,
                                 (1, opts.num_models),
                                 weight_tensor,
                                 target=None,
                                 max_row=max_row,
                                 params=full.params)

    else:

        preview = None

    return GaborModels(full, par_mini, one_mini, preview)

######################################################################

def main():
    
    opts = get_options()
    

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
                                      name='cur_error')

    max_row = tf.placeholder(tf.int32, shape=(), name='max_row')
    
    weight_tensor = tf.constant(weight_image)

    g = setup_models(opts, x, y,
                     weight_tensor,
                     target_tensor,
                     max_row)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    cur_approx = np.zeros_like(input_image)
    cur_error = input_image - cur_approx
    cur_con_losses = 0.0

    all_params = np.zeros((opts.num_models, GABOR_NUM_PARAMS), dtype=np.float32)
    all_approx = np.zeros((opts.num_models,) + input_image.shape, dtype=np.float32)
    all_con_loss = np.zeros(opts.num_models, dtype=np.float32)

    iteration = 0

    ginit = tf.global_variables_initializer()

    # prevent any additions to graph (they cause slowdowns!)
    tf.get_default_graph().finalize()

    prev_best_loss = None


    with tf.Session() as sess:

        sess.run(ginit)

        if opts.input is not None:
            if not os.path.isfile(opts.input):
                
                print("warning: can't load weights from", opts.input)
                
            else:
                
                foo = np.genfromtxt(opts.input, dtype=np.float32, delimiter=',')
                nfoo = len(foo)

                nfoo -= nfoo % opts.mini_ensemble_size
                nfoo = min(nfoo, opts.num_models)
                print(foo.shape, nfoo)
                
                all_params[:nfoo] = foo[:nfoo]

                g.full.params.load(all_params[None,:,:], sess)
                
                fetches = dict(gabor=g.full.gabor,
                               gabor_sum=g.full.gabor_sum,
                               err_loss=g.full.err_loss,
                               con_losses=g.full.con_loss_per_batch)
                
                cur_error = input_image
                feed_dict = {target_tensor: cur_error,
                             max_row: nfoo}
                
                results = sess.run(fetches, feed_dict)

                cur_approx = results['gabor_sum'][0]
                cur_error = input_image - cur_approx
                
                all_approx[:nfoo] = results['gabor'][0, :nfoo]
                all_con_loss[:nfoo] = results['con_losses'][:nfoo]

                prev_best_loss = results['err_loss'] + all_con_loss[:nfoo].sum()

                print('loaded {} models from {}; current loss is {}'.format(
                    nfoo, opts.input, prev_best_loss))

                iteration = nfoo

        if opts.preview_size:
            preview_stuff = (sess, g.preview, max_row)
        else:
            preview_stuff = None

        while True:

            sess.run(ginit)

            midx = iteration * opts.mini_ensemble_size
            is_replace = (midx >= opts.num_models)

            if (is_replace and
                (midx - opts.num_models) % opts.full_every == 0):

                print('performing full optimization!')
                print('  previous best loss was {}'.format(prev_best_loss))

                rparams = randomize(all_params, opts.rstdev)
                g.full.params.load(rparams[None,:,:], sess)
                
                cur_error = input_image
                feed_dict = {target_tensor: cur_error,
                             max_row: opts.num_models}

                fetches = dict(gabor=g.full.gabor,
                               gabor_sum=g.full.gabor_sum,
                               params=g.full.params,
                               loss=g.full.loss,
                               train_op=g.full.train_op,
                               con_losses=g.full.con_loss_per_batch)

                for i in range(opts.full_iter):

                    results = sess.run(fetches, feed_dict)
                    
                    if ((i+1) % 1000 == 0 and (i+1) < opts.full_iter):
                        
                        print('  loss at iter {:6d} is {}'.format(
                            i+1, results['loss']))

                        snapshot(results['gabor_sum'][0],
                                 results['gabor_sum'][0],
                                 input_image, weight_image,
                                 iteration, i, opts.label_snapshot,
                                 preview_stuff)

                        
                print('  new final loss is now  {}'.format(results['loss']))

                snapshot(results['gabor_sum'][0],
                         results['gabor_sum'][0],
                         input_image, weight_image,
                         iteration, opts.full_iter,
                         opts.label_snapshot,
                         preview_stuff)
                
                if results['loss'] < prev_best_loss:
                    all_params = results['params'][0]
                    all_approx = results['gabor'][0]
                    all_con_loss = results['con_losses']
                    prev_best_loss = results['loss']
                    
                    if opts.output is not None:
                        np.savetxt(opts.output,
                                   all_params[:min(iteration, opts.num_models)],
                                   fmt='%f', delimiter=',')
                    
                print()

            if is_replace:
                
                idx = np.arange(opts.num_models)
                np.random.shuffle(idx)

                models = idx[-opts.mini_ensemble_size:]
                idx = idx[:-opts.mini_ensemble_size]
                
                cur_approx = all_approx[idx].sum(axis=0)
                cur_con_losses = all_con_loss[idx].sum()
                cur_error = input_image - cur_approx
                
                print('replacing models', models)
                
            else:
                models = np.arange(opts.mini_ensemble_size) + midx
                print('training models', models)
                
            set_pvalues = is_replace or opts.lambda_err

            if set_pvalues:

                pvalues = sess.run(g.par_mini.params)
                
                if opts.lambda_err:
                    sample_uvs = sample_weighted_error(x, y, cur_error, weight_image,
                                              opts.lambda_err,
                                              (opts.num_parallel, opts.mini_ensemble_size))

                    pvalues[:, :, :2] = sample_uvs

                if is_replace:
                    pvalues[0, :, :] = randomize(all_params[models],
                                                 opts.rstdev)
                
                g.par_mini.params.load(pvalues, sess)

            fetches = dict(params=g.par_mini.params,
                           loss=g.par_mini.loss,
                           losses=g.par_mini.loss_per_fit,
                           con_loss_per_fit=g.par_mini.con_loss_per_fit,
                           con_loss_per_batch=g.par_mini.con_loss_per_batch,
                           con_loss=g.par_mini.con_loss,
                           train_op=g.par_mini.train_op,
                           best_loss=g.par_mini.losses_min,
                           best_params=g.par_mini.params_min)
            
            feed_dict = {target_tensor: cur_error}
          
            for i in range(opts.max_iter):
                results = sess.run(fetches, feed_dict)
                    

            best_loss = results['best_loss'] + cur_con_losses
            best_params = results['best_params']

            print('  best loss so far is', best_loss)

            g.one_mini.params.load(best_params[None,:], sess)

            fetches = dict(params=g.one_mini.params,
                           con_loss_per_batch=g.one_mini.con_loss_per_batch,
                           loss=g.one_mini.loss,
                           train_op=g.one_mini.train_op,
                           gabor=g.one_mini.gabor)

            for i in range(opts.refine):
                results = sess.run(fetches, feed_dict)

            new_loss = cur_con_losses + results['loss']
            print('  post-refine loss is', new_loss)

            if (is_replace and
                prev_best_loss is not None and
                new_loss >= prev_best_loss):
                
                print('  not better than', prev_best_loss, 'skipping update')
                print()
                
            else:

                if opts.output is not None:
                    np.savetxt(opts.output,
                               all_params[:min(iteration, opts.num_models)],
                               fmt='%f', delimiter=',')

                prev_best_loss = new_loss

                assert(results['params'].shape == (1, opts.mini_ensemble_size, 8))

                exp_shape = (1, opts.mini_ensemble_size) + input_image.shape

                assert(results['gabor'].shape == exp_shape)
                assert(results['con_loss_per_batch'].shape == (opts.mini_ensemble_size,))
                
                all_params[models] = results['params'][0,:]
                all_approx[models] = results['gabor'][0,:]
                all_con_loss[models] = results['con_loss_per_batch']

                cur_con_losses += results['con_loss_per_batch'].sum()
                cur_approx += all_approx[models].sum(axis=0)

                outfile = 'out{:04d}.png'.format(iteration+1)

                cur_error = input_image - cur_approx

                if preview_stuff is not None:
                    g.full.params.load(all_params[None,:,:], sess)

                foo = results['gabor'][0].sum(axis=0)
                
                snapshot(foo,
                         cur_approx, input_image, weight_image,
                         iteration, None, opts.label_snapshot,
                         preview_stuff)

                print()

            iteration += 1
            
if __name__ == '__main__':
    main()

