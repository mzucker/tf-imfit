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
    [ 0, 2] ])

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

    parser.add_argument('-l', '--lambda', type=float,
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

def make_gabor_tensor(x, y, params):

    clipped_params = tf.maximum(params, GABOR_RANGE[:,0])
    clipped_params = tf.minimum(params, GABOR_RANGE[:,1])

    u = clipped_params[GABOR_PARAM_U]
    v = clipped_params[GABOR_PARAM_V]
    r = clipped_params[GABOR_PARAM_R]
    p = clipped_params[GABOR_PARAM_P]
    l = clipped_params[GABOR_PARAM_L]
    t_clipme = clipped_params[GABOR_PARAM_T]
    s_clipme = clipped_params[GABOR_PARAM_S]
    h = clipped_params[GABOR_PARAM_H]

    s = tf.maximum(l/32., tf.minimum(l/2, s_clipme))
    t = tf.maximum(s, tf.minimum(s*8, t_clipme))

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

def main():
    
    opts = get_options()

    input_image = open_grayscale(opts.image, opts.max_size)
    weight_image = open_grayscale(opts.weights, opts.max_size)

    if weight_image is not None:
        assert weight_image.size == input_image.size

    h, w = input_image.shape
    hwmax = max(h, w)

    px = 2.0 / hwmax

    x = (np.arange(w, dtype=np.float32) - 0.5*(w) + 0.5) * px
    y = (np.arange(h, dtype=np.float32) - 0.5*(h) + 0.5) * px

    x = tf.constant(x.reshape(1, -1))
    y = tf.constant(y.reshape(-1, 1))

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
        diff = tf.constant(weight_image) * diff

    diffsqr = 0.5*diff**2

    loss = tf.reduce_mean(diffsqr, name='loss')

    with tf.variable_scope('imfit_optimizer'):
        opt = tf.train.AdamOptimizer(learning_rate=1e-5)
        train_op = opt.minimize(loss)

    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 'imfit_optimizer')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    cur_error = input_image
    
    with tf.Session() as sess:

        for model in range(opts.num_models):

            print('training model {}/{}...'.format(model+1, opts.num_models))

            best_params = None
            best_loss = None
            best_fit_iter = None

            fetches = dict(params=cparams,
                           loss=loss,
                           train_op=train_op)

            feed_dict = {cur_error_tensor: cur_error}
            
            for fit in range(opts.num_fits):

                print('  fit {}/{}...'.format(fit+1, opts.num_fits))
                
                sess.run(tf.global_variables_initializer())

                for i in range(opts.max_iter):

                    results = sess.run(fetches, feed_dict)
                    
                    if best_loss is None or results['loss'] < best_loss:
                        best_loss = results['loss']
                        best_params = results['params']
                        best_fit_iter = fit
                        print('    best loss is now {}'.format(best_loss))

            sess.run(tf.global_variables_initializer())
            sess.run(tf.assign(params, best_params))

            print('  refining solution from fit #{}'.format(best_fit_iter+1))
                        
            for refine in range(opts.refine):
                results = sess.run(fetches, feed_dict)
                if results['loss'] < best_loss:
                    best_loss = results['loss']
                    best_params = results['params']
                    print('    best loss is now {}'.format(best_loss))

            sys.exit(0)

        
if __name__ == '__main__':
    main()

