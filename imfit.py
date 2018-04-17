from __future__ import print_function
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

# TODO: separate constraint loss from approximation loss
# and don't compare sums of both iteration-to-iteration

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
                        default=100)

    parser.add_argument('-m', '--max-iter', type=int, metavar='N',
                        help='maximum # of iterations per trial fit',
                        default=10)

    parser.add_argument('-r', '--refine', type=int, metavar='N',
                        help='maximum # of iterations for final fit',
                        default=100)

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

    parser.add_argument('-R', '--replace-iter', type=int,
                        metavar='N',
                        help='maximum # of iterations for replacement')

    parser.add_argument('-l', '--lambda-err', type=float,
                        metavar='L',
                        help='weight on multinomial sampling of error map',
                        default=10.0)
    
    parser.add_argument('-J', '--joint-every', type=int, metavar='N',
                        help='perform joint optimization after every N models')

    parser.add_argument('-j', '--joint-iter', type=int, metavar='N',
                        help='maximum # of iterations for joint optimization',
                        default=100)

    args = parser.parse_args()

    if args.joint_every is None:
        args.joint_every = args.num_models

    return args

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

    def __init__(self, x, y, nparams, weight,
                 cur_error_tensor,
                 separate_loss=False,
                 learning_rate=1e-4,
                 initializer=None):

        if initializer is None:
            gmin = GABOR_RANGE[:,0].reshape(1,GABOR_NUM_PARAMS)
            gmax = GABOR_RANGE[:,1].reshape(1,GABOR_NUM_PARAMS)
            initializer = tf.random_uniform_initializer(minval=gmin,
                                                        maxval=gmax,
                                                        dtype=tf.float32)
            
        self.params = tf.get_variable('params',
                                      shape=(nparams, GABOR_NUM_PARAMS),
                                      dtype=tf.float32,
                                      initializer=initializer)

        l = self.params[:,GABOR_PARAM_L]
        s = self.params[:,GABOR_PARAM_S]
        t = self.params[:,GABOR_PARAM_T]

        c0 = s - l/32
        c1 = l/2 - s
        c2 = t - s
        c3 = 8*s - t

        cbounds = []
        for i  in range(GABOR_PARAM_P, GABOR_NUM_PARAMS):
            lo, hi = GABOR_RANGE[i]
            var = self.params[:,i]
            if lo != -np.pi:
                cbounds.append( var - lo )
            if hi != np.pi:
                cbounds.append( hi - var)

        self.cfuncs = tf.stack( [c0, c1, c2, c3] + cbounds,
                                axis=1, name='cfuncs' )

        params_bcast = self.params[:,:,None,None]

        u = params_bcast[:,GABOR_PARAM_U]
        v = params_bcast[:,GABOR_PARAM_V]
        r = params_bcast[:,GABOR_PARAM_R]
        p = params_bcast[:,GABOR_PARAM_P]
        l = params_bcast[:,GABOR_PARAM_L]
        s = params_bcast[:,GABOR_PARAM_S]
        t = params_bcast[:,GABOR_PARAM_T]
        h = params_bcast[:,GABOR_PARAM_H]

        cr = tf.cos(r)
        sr = tf.sin(r)

        f = np.float32(2*np.pi) / l

        s2 = s*s
        t2 = t*t

        xp = x-u
        yp = y-v

        b1 =  cr*xp + sr*yp
        b2 = -sr*xp + cr*yp

        b12 = b1*b1
        b22 = b2*b2

        w = tf.exp(-b12/(2*s2) - b22/(2*t2))

        k = f*b1 +  p
        ck = tf.cos(k)

        self.gabor = tf.identity(h * w * ck, name='gabor')

        cviol = tf.minimum(self.cfuncs, 0)
        cviol = cviol**2

        if separate_loss:

            self.diff = (cur_error_tensor - self.gabor) * weight
        
            diffsqr = 0.5*self.diff**2
        
            self.losses = ( tf.reduce_mean(diffsqr, axis=(1,2)) +
                            tf.reduce_mean(cviol, axis=1) )

            self.losses_argmin = tf.argmin(self.losses)

            self.losses_min = self.losses[self.losses_argmin]
            self.params_min = self.params[self.losses_argmin]

            self.loss = tf.reduce_mean(self.losses)

        else:

            self.diff = (cur_error_tensor -
                         tf.reduce_sum(self.gabor, axis=0)[None,:,:]) * weight

            print('diff is', self.diff)

            diffsqr = 0.5*self.diff**2

            self.loss = tf.reduce_mean(diffsqr) + tf.reduce_mean(cviol)

        with tf.variable_scope('imfit_optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.opt.minimize(self.loss)

######################################################################

def sample_error(x, y, err, weight, lambda_err, count):

    if weight is not None:
        err = weight * err

    p = lambda_err*err**2
    p = np.exp(p - p.max())
    psum = p.sum()
    
    h, w = p.shape

    p = p.flatten()
        
    assert y.shape == (1, h, 1)
    assert x.shape == (1, 1, w)

    prefix_sum = np.cumsum(p)
    r = np.random.random(count) * prefix_sum[-1]

    idx = np.searchsorted(prefix_sum, r)
    
    row = idx / w
    col = idx % w

    assert row.min() >= 0 and row.max() < h
    assert col.min() >= 0 and col.max() < w

    u = x.flatten()[col]
    v = y.flatten()[row]

    uv = np.hstack( ( u.reshape(-1, 1), v.reshape(-1, 1) ) )

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

def main():
    
    opts = get_options()

    input_image = open_grayscale(opts.image, opts.max_size)
    weight_image = open_grayscale(opts.weights, opts.max_size)

    if weight_image is not None:
        assert weight_image.size == input_image.size

    # move to -1, 1 range for input image
    input_image = input_image * 2 - 1

    h, w = input_image.shape
    hwmax = max(h, w)

    px = 2.0 / hwmax

    x = (np.arange(w, dtype=np.float32) - 0.5*(w) + 0.5) * px
    y = (np.arange(h, dtype=np.float32) - 0.5*(h) + 0.5) * px

    # shape is num_fits x height x width
    x = x.reshape(1,  1, -1)
    y = y.reshape(1, -1,  1)

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
    
    cur_error_tensor = tf.placeholder(tf.float32,
                                      shape=(1,) + input_image.shape,
                                      name='cur_error')

    if weight_image is not None:
        wimg = tf.constant(weight_image)
    else:
        wimg = 1.0

    with tf.variable_scope('small'):
        g_small = GaborModel(x, y, 1, wimg,
                             cur_error_tensor)

    with tf.variable_scope('big'):
        g_big = GaborModel(x, y, opts.num_fits, wimg,
                           cur_error_tensor, separate_loss=True)

    with tf.variable_scope('joint'):
        g_joint = GaborModel(x, y, opts.num_models, wimg,
                             cur_error_tensor,
                             learning_rate=1e-5,
                             initializer=tf.zeros_initializer())
        
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 '.*/imfit_optimizer')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    cur_approx = np.zeros_like(input_image)
    cur_error = input_image - cur_approx

    all_params = np.zeros((opts.num_models, GABOR_NUM_PARAMS), dtype=np.float32)
    all_approx = np.zeros((opts.num_models,) + input_image.shape, dtype=np.float32)

    iteration = 0

    ginit = tf.global_variables_initializer()

    # prevent any additions to graph (they cause slowdowns!)
    tf.get_default_graph().finalize()

    prev_best_loss = None

    
    with tf.Session() as sess:

        while True:

            sess.run(ginit)
            
            is_replace = (iteration >= opts.num_models)

            if (is_replace and
                (iteration - opts.num_models) % opts.joint_every == 0):

                print('performing joint optimization!')
                print('  loss was', prev_best_loss)

                g_joint.params.load(all_params, sess)
                
                cur_error = input_image
                feed_dict = {cur_error_tensor: cur_error[None,:,:]}

                fetches = dict(gabor=g_joint.gabor,
                               params=g_joint.params,
                               loss=g_joint.loss,
                               train_op=g_joint.train_op)

                for i in range(opts.joint_iter):
                    results = sess.run(fetches, feed_dict)

                print('  loss is ', results['loss'])

                all_params = results['params']
                all_approx = results['gabor']

                print()

            if is_replace:
                model = np.random.randint(opts.num_models)
                idx = np.hstack((np.arange(0,model),
                                 np.arange(model+1,opts.num_models)))
                cur_approx = all_approx[idx].sum(axis=0)
                cur_error = input_image - cur_approx
                print('replacing model {}/{}'.format(model+1, opts.num_models))
            else:
                model = iteration
                print('training model {}/{}'.format(model+1, opts.num_models))
                
            set_pvalues = is_replace or opts.lambda_err

            if set_pvalues:

                pvalues = sess.run(g_big.params)
                
                if opts.lambda_err:
                    sample_uvs = sample_error(x, y, cur_error, weight_image,
                                              opts.lambda_err, opts.num_fits)
                    pvalues[:, :2] = sample_uvs

                if is_replace:
                    pvalues[0, :] = all_params[model]
                
                g_big.params.load(pvalues, sess)

            fetches = dict(params=g_big.params,
                           loss=g_big.loss,
                           losses=g_big.losses,
                           train_op=g_big.train_op,
                           best_loss=g_big.losses_min,
                           best_params=g_big.params_min)
            
            feed_dict = {cur_error_tensor: cur_error[None,:,:]}
          
            for i in range(opts.max_iter):
                results = sess.run(fetches, feed_dict)

            best_loss = results['best_loss']
            best_params = results['best_params']

            print('  best loss so far is', best_loss)

            g_small.params.load(best_params[None,:], sess)

            fetches = dict(params=g_small.params,
                           loss=g_small.loss,
                           train_op=g_small.train_op,
                           gabor=g_small.gabor)

            for i in range(opts.refine):
                results = sess.run(fetches, feed_dict)

            print('  refined loss is    ', results['loss'])

            if (is_replace and
                prev_best_loss is not None and
                results['loss'] >= prev_best_loss):
                
                print('  not better than', prev_best_loss, 'skipping update')
                print()
                
            else:

                prev_best_loss = results['loss']

                all_params[model] = results['params'][0]
                all_approx[model] = results['gabor'][0]

                cur_approx += all_approx[model]
                cur_error = input_image - cur_approx

                cur_abserr = np.abs(cur_error)
                if weight_image is not None:
                    cur_abserr = cur_abserr * weight_image

                out_img = np.hstack(( rescale(input_image, -1, 1),
                                      rescale(cur_approx, -1, 1),
                                      rescale(cur_abserr, 0, 1.0) ))

                out_img = Image.fromarray(out_img, 'L')

                outfile = 'out{:04d}.png'.format(iteration)
                out_img.save(outfile)

                print('  wrote {}'.format(outfile))
                print()

            iteration += 1
            
if __name__ == '__main__':
    main()

