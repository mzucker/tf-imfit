from __future__ import print_function
import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

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
    [ -2, 2 ],
    [ -2, 2 ],
    [ 0, 2*np.pi ],
    [ 0, 2*np.pi ],
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
    
    args = parser.parse_args()

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

def make_gabor_tensor(x, y, params):

    params = tf.clip_by_value(params, GABOR_RANGE[:,0], GABOR_RANGE[:,1])

    u = params[GABOR_PARAM_U]
    v = params[GABOR_PARAM_V]
    r = params[GABOR_PARAM_R]
    p = params[GABOR_PARAM_P]
    l = params[GABOR_PARAM_L]
    s = params[GABOR_PARAM_S]
    t = params[GABOR_PARAM_T]
    h = params[GABOR_PARAM_H]

    s = tf.clip_by_value(s, l/32, l/8)
    t = tf.clip_by_value(t, s, 8*s)

    clipped_params = tf.stack( (u, v, r, p, l, t, s, h), axis=0,
                               name='clipped_params' )
    
    cr = tf.cos(r)
    sr = tf.sin(r)

    f = np.float32(2*np.pi) / l

    s2 = s*s
    t2 = t*t

    xp = x-u
    yp = y-v

    b1 = cr*xp + sr*yp
    b2 = -sr*xp + cr*yp

    b12 = b1*b1
    b22 = b2*b2

    w = tf.exp(-b12/(2*s2) - b22/(2*t2))

    k = f*b1 +  p
    ck = tf.cos(k)
    
    gabor = tf.identity(h * w * ck, name='gabor')

    return clipped_params, gabor

######################################################################

def sample_error(x, y, err, weight, lambda_err, count):

    if weight is not None:
        err = weight * err

    p = lambda_err*err**2
    p = np.exp(p - p.max())
    psum = p.sum()
    
    h, w = p.shape

    p = p.flatten()

        
    assert y.shape == (h, 1)
    assert x.shape == (1, w)

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

    px = x[0,1]-x[0,0]
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

    x = x.reshape(1, -1)
    y = y.reshape(-1, 1)

    GABOR_RANGE[GABOR_PARAM_L, 0] = 2.5*px
    GABOR_RANGE[GABOR_PARAM_T, 0] = px
    GABOR_RANGE[GABOR_PARAM_S, 0] = px
    
    params = tf.get_variable('params', shape=8, dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(
                                 minval=GABOR_RANGE[:,0],
                                 maxval=GABOR_RANGE[:,1],
                                 dtype=tf.float32))
                             
    cparams, gabor = make_gabor_tensor(x, y, params)

    cur_error_tensor = tf.placeholder(tf.float32, shape=input_image.shape,
                                      name='cur_error')
    
    diff = cur_error_tensor - gabor

    if weight_image is not None:
        diff = tf.constant(weight_image**2) * diff

    diffsqr = 0.5*diff**2

    loss = tf.reduce_mean(diffsqr, name='loss')

    with tf.variable_scope('imfit_optimizer'):
        #opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = opt.minimize(loss)

    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 'imfit_optimizer')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    cur_approx = np.zeros_like(input_image)
    cur_error = input_image - cur_approx

    all_params = []

    # prevent any additions to graph (they cause slowdowns!)
    tf.get_default_graph().finalize()
    
    with tf.Session() as sess:

        for model in range(opts.num_models):

            print('training model {}/{}'.format(model+1, opts.num_models))

            best_params = None
            best_loss = None
            best_fit_iter = None

            fetches = dict(params=params,
                           cparams=cparams,
                           loss=loss,
                           train_op=train_op)

            feed_dict = {cur_error_tensor: cur_error}

            if opts.lambda_err:
                sample_uvs = sample_error(x, y, cur_error, weight_image,
                                          opts.lambda_err, opts.num_fits)

        
            for fit in range(opts.num_fits):

                sess.run(params.initializer)

                if opts.lambda_err:
                    #pvalues = sess.run(params)
                    #pvalues[:2] = sample_uvs[fit]
                    #params.load(pvalues, sess)
                    pass

                sess.run([var.initializer for var in opt_vars])
                    
                outchar = '.'

                for i in range(opts.max_iter):

                    results = sess.run(fetches, feed_dict)
                    #if i == 0:
                    #    print(results['cparams'])
                    
                    if best_loss is None or results['loss'] < best_loss:
                        best_loss = results['loss']
                        best_params = results['params']
                        best_fit_iter = fit
                        outchar = '*'

                sys.stdout.write(outchar)
                sys.stdout.flush()
                        

            sess.run([var.initializer for var in opt_vars])
            params.load(best_params, sess)

            print()
            print('  refining solution from fit #{}'.format(best_fit_iter+1))
                        
            for refine in range(opts.refine):
                results = sess.run(fetches, feed_dict)
                if results['loss'] < best_loss:
                    best_loss = results['loss']
                    best_params = results['params']

            print('  loss after model {} is {}'.format(model+1, best_loss))

            all_params.append(best_params)
            params.load(best_params, sess)

            fetches = dict(gabor=gabor, diff=diff)

            results = sess.run(fetches, feed_dict)

            cur_approx += results['gabor']
            cur_error = input_image - cur_approx

            cur_abserr = np.abs(cur_error)
            if weight_image is not None:
                cur_abserr = cur_abserr * weight_image


            out_img = np.hstack(( rescale(input_image, -1, 1),
                                  rescale(cur_approx, -1, 1),
                                  rescale(cur_abserr, 0, 1.0) ))
                
            out_img = Image.fromarray(out_img, 'L')

            outfile = 'out{:04d}.png'.format(model)
            out_img.save(outfile)
            
            print('  wrote {}'.format(outfile))
            print()
        
if __name__ == '__main__':
    main()

