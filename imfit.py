from __future__ import print_function
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

# DONE: replace multiple parameters at once (doesn't help)
# TODO: large-size preview
# TODO: load/save parameter sets

######################################################################

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

def get_options():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('image', type=argparse.FileType('r'),
                        metavar='IMAGE.png',
                        help='image to approximate')

    parser.add_argument('-n', '--num-models', type=int, metavar='N',
                        help='number of models to fit',
                        default=64)

    parser.add_argument('-f', '--num-fits', type=int, metavar='N',
                        help='number of random guesses per model',
                        default=200)

    parser.add_argument('-m', '--max-iter', type=int, metavar='N',
                        help='maximum # of iterations per trial fit',
                        default=10)

    parser.add_argument('-r', '--refine', type=int, metavar='N',
                        help='maximum # of iterations for final fit',
                        default=100)

    parser.add_argument('-R', '--rstdev', type=float, metavar='R',
                        help='amount to randomize models when re-fitting',
                        default=0.001)

    parser.add_argument('-s', '--max-size', type=int, metavar='N',
                        help='maximum size of image to load',
                        default=32)

    parser.add_argument('-w', '--weights', type=argparse.FileType('r'),
                        metavar='WEIGHTS.png',
                        help='load weights from file',
                        default=512)

    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                        metavar='PARAMFILE.txt',
                        help='read input params from file')

    parser.add_argument('-o', '--output', type=argparse.FileType('w'),
                        metavar='PARAMFILE.txt',
                        help='write input params to file')

    parser.add_argument('-l', '--lambda-err', type=float,
                        metavar='L',
                        help='weight on multinomial sampling of error map',
                        default=2.0)
    
    parser.add_argument('-J', '--joint-every', type=int, metavar='N',
                        help='perform joint optimization after every N models')

    parser.add_argument('-j', '--joint-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=10000)

    parser.add_argument('-S', '--single-snapshot', action='store_true',
                        help='prevent labeling snapshot images')

    parser.add_argument('-b', '--batch-size', type=int, metavar='N',
                        help='models to update per iteration',
                        default=1)
    
    opts = parser.parse_args()

    if opts.joint_every is None:
        opts.joint_every = opts.num_models

    assert opts.num_models % opts.batch_size == 0
    assert opts.joint_every % opts.batch_size == 0 
       
    return opts

######################################################################

def open_grayscale(handle, max_size):

    if handle is None:
        return None

    image = Image.open(handle)

    if image.mode != 'L':
        print('converting {} to grayscale'.format(handle.name))
        image = image.convert('L')

    w, h = image.size

    
    if max(w, h) > max_size:
        
        if w > h:
            wnew = max_size
            hnew = int(round(float(h) * max_size / w))
        else:
            wnew = int(round(float(w) * max_size / h))
            hnew = max_size
            
        image = image.resize((wnew, hnew), resample=Image.LANCZOS)
        w, h = wnew, hnew

    print('{} is {}x{}'.format(handle.name, w, h))

    image = np.array(image).astype(np.float32) / 255.
    
    return image

######################################################################

def mix(a, b, u):
    return a + u*(b-a)

######################################################################

class GaborModel(object):

    def __init__(self, x, y, minishape, weight,
                 cur_error_tensor,
                 separate_loss=False,
                 learning_rate=1e-4,
                 initializer=None):

        num_parallel, num_joint = minishape

        if initializer is None:
            gmin = GABOR_RANGE[:,0].reshape(1,1,GABOR_NUM_PARAMS)
            gmax = GABOR_RANGE[:,1].reshape(1,1,GABOR_NUM_PARAMS)
            initializer = tf.random_uniform_initializer(minval=gmin,
                                                        maxval=gmax,
                                                        dtype=tf.float32)

        # f x b x 8
        self.params = tf.get_variable(
            'params',
            shape=(num_parallel, num_joint, GABOR_NUM_PARAMS),
            dtype=tf.float32,
            initializer=initializer)

        l = self.params[:,:,GABOR_PARAM_L]
        s = self.params[:,:,GABOR_PARAM_S]
        t = self.params[:,:,GABOR_PARAM_T]

        c0 = s - l/32
        c1 = l/2 - s
        c2 = t - s
        c3 = 8*s - t

        cbounds = []
        for i  in range(GABOR_PARAM_P, GABOR_NUM_PARAMS):
            lo, hi = GABOR_RANGE[i]
            var = self.params[:,:,i]
            if lo != -np.pi:
                cbounds.append( var - lo )
            if hi != np.pi:
                cbounds.append( hi - var)

        # f x b x k
        self.cfuncs = tf.stack( [c0, c1, c2, c3] + cbounds,
                                axis=2, name='cfuncs' )

        # f x b x 8 x 1 x 1
        params_bcast = self.params[:,:,:,None,None]

        # f x b x 1 x 1
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

        # f x b x 1 x w
        xp = x-u

        # f x b x h x 1
        yp = y-v

        # f x b x h x w
        b1 =  cr*xp + sr*yp
        b2 = -sr*xp + cr*yp

        b12 = b1*b1
        b22 = b2*b2

        w = tf.exp(-b12/(2*s2) - b22/(2*t2))

        k = f*b1 +  p
        ck = tf.cos(k)

        self.gabor = tf.identity(h * w * ck, name='gabor')

        # f x h x w
        self.gabor_sum = tf.reduce_sum(self.gabor, axis=1, name='gabor_sum')

        # f x h x w
        self.diff = tf.multiply((cur_error_tensor - self.gabor_sum),
                                weight, name='diff')

        diffsqr = 0.5*self.diff**2

        # f
        self.e_loss_per_fit = tf.reduce_mean(diffsqr, axis=(1,2),
                                             name='e_loss_per_fit')

        # f x b x k
        cviol = tf.pow(tf.minimum(self.cfuncs, 0), 2, name='cviol')
        
        # f x b
        self.c_loss_all = tf.reduce_sum(cviol, axis=2, name='c_loss_all')

        # f
        self.c_loss_per_fit = tf.reduce_sum(self.c_loss_all, axis=1,
                                            name='c_loss_per_fit')

        # b
        self.c_loss_per_batch = tf.reduce_mean(self.c_loss_all, axis=0,
                                               name='c_loss_per_batch')

        # f
        self.loss_per_fit = tf.add(self.c_loss_per_fit,
                                   self.e_loss_per_fit,
                                   name='loss_per_fit')

        lpf_argmin = tf.argmin(self.loss_per_fit)

        # scalar
        self.losses_min = self.loss_per_fit[lpf_argmin]

        # b x 8
        self.params_min = self.params[lpf_argmin]

        # scalars
        self.e_loss = tf.reduce_mean(self.e_loss_per_fit, name='e_loss')
        self.c_loss = tf.reduce_mean(self.c_loss_per_fit, name='c_loss')
        self.loss = self.e_loss + self.c_loss

        with tf.variable_scope('imfit_optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.opt.minimize(self.loss)

######################################################################

def sample_error(x, y, err, weight, lambda_err,
                 minishape):

    num_parallel, num_joint = minishape

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

def snapshot(cur_approx, input_image, 
             weight_image,
             iteration, joint_iteration,
             single_file):

    if single_file:
        outfile = 'out.png'
    elif joint_iteration is None:
        outfile = 'out{:04d}_single.png'.format(iteration+1)
    else:
        outfile = 'out{:04d}_{:06d}.png'.format(
            iteration+1, joint_iteration+1)

    cur_abserr = np.abs(cur_approx - input_image)
    cur_abserr = cur_abserr * weight_image

    out_img = np.hstack(( rescale(input_image, -1, 1),
                          rescale(cur_approx, -1, 1),
                          rescale(cur_abserr, 0, 1.0) ))

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

    h, w = input_image.shape
    hwmax = max(h, w)

    px = 2.0 / hwmax

    x = (np.arange(w, dtype=np.float32) - 0.5*(w) + 0.5) * px
    y = (np.arange(h, dtype=np.float32) - 0.5*(h) + 0.5) * px

    # shape broadcastable to 1 x 1 x h x w
    x = x.reshape(1, 1,  1, -1)
    y = y.reshape(1, 1, -1,  1)

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
    
    cur_error_tensor = tf.placeholder(tf.float32,
                                      shape=input_image.shape,
                                      name='cur_error')

    wimg = tf.constant(weight_image)

    with tf.variable_scope('joint'):
        g_joint = GaborModel(x, y, (1, opts.num_models), wimg,
                             cur_error_tensor,
                             initializer=tf.zeros_initializer())
    

    with tf.variable_scope('big'):
        g_big = GaborModel(x, y, (opts.num_fits, opts.batch_size), wimg,
                           cur_error_tensor)

    with tf.variable_scope('small'):
        g_small = GaborModel(x, y, (1, opts.batch_size), wimg,
                             cur_error_tensor)
        
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 '.*/imfit_optimizer')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    cur_approx = np.zeros_like(input_image)
    cur_error = input_image - cur_approx
    cur_c_losses = 0.0

    all_params = np.zeros((opts.num_models, GABOR_NUM_PARAMS), dtype=np.float32)
    all_approx = np.zeros((opts.num_models,) + input_image.shape, dtype=np.float32)
    all_c_loss = np.zeros(opts.num_models, dtype=np.float32)

    iteration = 0

    ginit = tf.global_variables_initializer()

    # prevent any additions to graph (they cause slowdowns!)
    tf.get_default_graph().finalize()

    prev_best_loss = None


    with tf.Session() as sess:

        while True:

            sess.run(ginit)

            midx = iteration * opts.batch_size
            is_replace = (midx >= opts.num_models)

            if (is_replace and
                (midx - opts.num_models) % opts.joint_every == 0):

                print('performing joint optimization!')
                print('  previous best loss was {}'.format(prev_best_loss))

                rparams = randomize(all_params, opts.rstdev)
                g_joint.params.load(rparams[None,:,:], sess)
                
                cur_error = input_image
                feed_dict = {cur_error_tensor: cur_error}

                fetches = dict(gabor=g_joint.gabor,
                               gabor_sum=g_joint.gabor_sum,
                               params=g_joint.params,
                               loss=g_joint.loss,
                               train_op=g_joint.train_op,
                               c_losses=g_joint.c_loss_per_batch)

                for i in range(opts.joint_iter):

                    results = sess.run(fetches, feed_dict)
                    
                    if ((i+1) % 1000 == 0 and (i+1) < opts.joint_iter):
                        
                        print('  loss at iter {:6d} is {}'.format(
                            i+1, results['loss']))

                        snapshot(results['gabor_sum'][0],
                                 input_image, weight_image,
                                 iteration, i, opts.single_snapshot)

                print('  new final loss is now  {}'.format(results['loss']))

                snapshot(results['gabor_sum'][0],
                         input_image, weight_image,
                         iteration, opts.joint_iter,
                         opts.single_snapshot)
                
                if results['loss'] < prev_best_loss:
                    all_params = results['params'][0]
                    all_approx = results['gabor'][0]
                    all_c_loss = results['c_losses']
                    prev_best_loss = results['loss']

                print()

            if is_replace:
                
                idx = np.arange(opts.num_models)
                np.random.shuffle(idx)

                models = idx[-opts.batch_size:]
                idx = idx[:-opts.batch_size]
                
                cur_approx = all_approx[idx].sum(axis=0)
                cur_c_losses = all_c_loss[idx].sum()
                cur_error = input_image - cur_approx
                
                print('replacing models', models)
                
            else:
                models = np.arange(opts.batch_size) + midx
                print('training models', models)
                
            set_pvalues = is_replace or opts.lambda_err

            if set_pvalues:

                pvalues = sess.run(g_big.params)
                
                if opts.lambda_err:
                    sample_uvs = sample_error(x, y, cur_error, weight_image,
                                              opts.lambda_err,
                                              (opts.num_fits, opts.batch_size))

                    pvalues[:, :, :2] = sample_uvs

                if is_replace:
                    pvalues[0, :, :] = randomize(all_params[models],
                                                 opts.rstdev)
                
                g_big.params.load(pvalues, sess)

            fetches = dict(params=g_big.params,
                           loss=g_big.loss,
                           losses=g_big.loss_per_fit,
                           c_loss_per_fit=g_big.c_loss_per_fit,
                           c_loss_per_batch=g_big.c_loss_per_batch,
                           c_loss=g_big.c_loss,
                           train_op=g_big.train_op,
                           best_loss=g_big.losses_min,
                           best_params=g_big.params_min)
            
            feed_dict = {cur_error_tensor: cur_error}
          
            for i in range(opts.max_iter):
                results = sess.run(fetches, feed_dict)
                    

            best_loss = results['best_loss'] + cur_c_losses
            best_params = results['best_params']

            print('  best loss so far is', best_loss)

            g_small.params.load(best_params[None,:], sess)

            fetches = dict(params=g_small.params,
                           c_loss_per_batch=g_small.c_loss_per_batch,
                           loss=g_small.loss,
                           train_op=g_small.train_op,
                           gabor=g_small.gabor)

            for i in range(opts.refine):
                results = sess.run(fetches, feed_dict)

            new_loss = cur_c_losses + results['loss']
            print('  post-refine loss is', new_loss)

            if (is_replace and
                prev_best_loss is not None and
                new_loss >= prev_best_loss):
                
                print('  not better than', prev_best_loss, 'skipping update')
                print()
                
            else:

                prev_best_loss = new_loss

                assert(results['params'].shape == (1, opts.batch_size, 8))

                exp_shape = (1, opts.batch_size) + input_image.shape

                assert(results['gabor'].shape == exp_shape)
                assert(results['c_loss_per_batch'].shape == (opts.batch_size,))
                
                all_params[models] = results['params'][0,:]
                all_approx[models] = results['gabor'][0,:]
                all_c_loss[models] = results['c_loss_per_batch']

                cur_c_losses += results['c_loss_per_batch'].sum()
                cur_approx += all_approx[models].sum(axis=0)

                outfile = 'out{:04d}.png'.format(iteration+1)

                cur_error = input_image - cur_approx
                
                snapshot(cur_approx, input_image, weight_image,
                         iteration, None, opts.single_snapshot)

                print()

            iteration += 1
            
if __name__ == '__main__':
    main()

