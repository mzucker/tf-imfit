#!/usr/bin/env python

'''Script to encode the list of 8-tuples emitted by the imfit program
and format them for use in the shader. The order emitted by imfit is:

  u: center x-coordinate of Gabor basis function
  v: center y-coordinate
  r: rotation angle of Gabor basis function
  p: phase of basis function
  l: wavelength (inverse frequency) of Gabor function
  t: width perpendicular to sinusoidal component
  s: width along sinusoidal component
  h: amplitude

The eight numbers are rescaled and quantized into the range [0, 511]
and encoded into four floating-point numbers (uv, rp, lt, sh).

'''

import sys
import numpy as np

def wrap_twopi(f):
    while f < 0: f += 2*np.pi
    while f > 2*np.pi: f -= 2*np.pi
    return f

if len(sys.argv) != 2:
    print 'usage:', sys.argv[0], 'params.txt'
    sys.exit(0)

infile = open(sys.argv[1], 'r')



var_names = 'uvrpltsh'
tol = 1e-4

for line in infile:

    line = line.rstrip()
    if not line:
        break
    
    nums = np.array(map(float, line.split(',')))

    uvrp = 'vec4({},{},{},{})'.format(*nums[:4])
    ltsh = 'vec4({},{},{},{})'.format(*nums[4:])
        
    print '    k += gabor(p, {}, {});'.format(uvrp, ltsh)
