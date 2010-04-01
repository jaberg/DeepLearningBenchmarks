#!/usr/bin/env python2.5
from __future__ import absolute_import
import numpy as N
import sys
import time

# c: aa.cc

nin, nhid, nout, batchsize, niter = [int(a) for a in sys.argv[1:]]
lr = 0.01

rng = N.random.RandomState(342)

# declare data
x = (rng.rand(batchsize*niter, nin)-0.5) * 1.5
y = (rng.rand(batchsize*niter, nout)-0.5) * 1.5

# declare model weights
w = rng.rand(nin, nhid)
b = rng.randn(nhid) * 0.0
v = rng.rand(nhid, nout)
c = rng.randn(nout) * 0.0

t = time.time()
for i in xrange(niter):
    x_i = x[i*batchsize:(i+1)*batchsize]
    y_i = y[i*batchsize:(i+1)*batchsize]

    hidin = N.dot(x_i, w) + b

    hidout = N.tanh(hidin)

    outin = N.dot(hidout, v) + c
    outout = (N.tanh(outin)+1)/2.0

    g_outout = outout - y_i
    err = 0.5 * N.sum(g_outout**2)

    g_outin = g_outout * outout * (1.0 - outout)

    g_hidout = N.dot(g_outin, v.T)
    g_hidin = g_hidout * (1 - hidout**2)

    b -= lr * N.sum(g_hidin, axis=0)
    c -= lr * N.sum(g_outin, axis=0)
    w -= lr * N.dot(x_i.T, g_hidin)
    v -= lr * N.dot(hidout.T, g_outin)

total_time = time.time() - t
print 'mlp_%i_%i_%i\tnumpy{%i}\t%.2f' %(
        nin, nhid, nout, batchsize, niter*batchsize/total_time)

