#!/usr/bin/env python2.5
from __future__ import absolute_import
import numpy as N
import sys
import time

# c: aa.cc

nin, nout, batchsize, niter = [int(a) for a in sys.argv[1:]]
lr = 0.01

rng = N.random.RandomState(342)

# declare data
x = (rng.rand(batchsize*niter, nin)-0.5) * 1.5
y = (rng.rand(batchsize*niter, nout)-0.5) * 1.5

# declare model weights
w = rng.rand(nin, nout)
b = rng.randn(nout) * 0.0

t = time.time()
for i in xrange(niter):
    x_i = x[i*batchsize:(i+1)*batchsize]
    y_i = y[i*batchsize:(i+1)*batchsize]

    hidin = N.dot(x_i, w) + b

    hidout = (N.tanh(hidin)+1)/2.0 # sigmoid

    g_hidout = hidout - y_i
    err = 0.5 * N.sum(g_hidout**2)

    g_hidin = g_hidout * hidout * (1.0 - hidout)

    b -= lr * N.sum(g_hidin, axis=0)
    w -= lr * N.dot(x_i.T, g_hidin)

total_time = time.time() - t
print 'mlp_%i_%i\tnumpy{%i}\t%.2f' %(
        nin, nout, batchsize, niter*batchsize/total_time)

