#!/usr/bin/env python2.5
from __future__ import absolute_import
import numpy as np
import sys
import time
from theano.tensor import lscalar, dot, sum as tsum
from theano.tensor.nnet import sigmoid
from theano import shared, function, config

rng = np.random.RandomState(342)

def rand(*size):
    return np.asarray(rng.rand(*size), dtype=config.floatX)
def randn(*size):
    return np.asarray(rng.randn(*size), dtype=config.floatX)
def randint(size, high):
    return np.asarray(rng.randint(size=size, low=0, high=high), dtype='int32')
def zeros(*size):
    return np.zeros(size, dtype=config.floatX)

# c: aa.cc

nin, nout, batchsize, niter = [int(a) for a in sys.argv[1:]]
lr = 0.01

# declare data
data_x = shared(rand(batchsize*niter, nin))
si = lscalar()
nsi = lscalar()

# declare model weights
a = shared(zeros(nin))
w = shared(zeros(nin, nout))
b = shared(zeros(nout))

import theano.sandbox.rng_mrg
R = theano.sandbox.rng_mrg.MRG_RandomStreams()

def bern(x, size):
    return R.binomial(size=size, p=x, n=1, dtype=config.floatX)

pos_vis = data_x[si:si+nsi]

pos_hid = sigmoid(dot(pos_vis, w)+b)

neg_vis = sigmoid(dot(bern(pos_hid, (batchsize, nout)), w.T)+a)

neg_hid = sigmoid(dot(bern(neg_vis, (batchsize, nin)), w) + b)

new_a = a - lr * tsum(pos_vis - neg_vis, axis=0)
new_b = b - lr * tsum(pos_hid - neg_hid, axis=0)
new_w = w - lr * (dot(pos_vis.T, pos_hid) - dot(neg_vis.T, neg_hid))

f = function([si, nsi], [], updates={a:new_a, b:new_b, w:new_w})

t = time.time()
for i in xrange(niter):
    f(i*batchsize, batchsize)

print 'cd1 rbm_bernoulli %i_%i\ttheano{%s/%s/%i}\t%.2f' %(
        nin, nout, 
        config.device[0],
        ('float' if config.floatX == 'float32' else 'double'),
        batchsize, 
        niter*batchsize/(time.time() - t))


