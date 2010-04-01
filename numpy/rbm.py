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

# declare model weights
a = rng.randn(nin) * 0.0
w = rng.rand(nin, nout)
b = rng.randn(nout) * 0.0

def sigm(x): return (N.tanh(x)+1)/2

def bern(x): return N.random.binomial(p=x,n=1)

t = time.time()
for i in xrange(niter):
    pos_vis = x[i*batchsize:(i+1)*batchsize]

    pos_hid = sigm(N.dot(pos_vis, w)+b)

    neg_vis = sigm(N.dot(bern(pos_hid), w.T)+a)

    neg_hid = sigm(N.dot(bern(neg_vis), w) + b)

    a += lr * N.sum(pos_vis - neg_vis, axis=0)
    b -= lr * N.sum(pos_hid - neg_hid, axis=0)
    w -= lr * (N.dot(pos_vis.T, pos_hid) - N.dot(neg_vis.T, neg_hid))

total_time = time.time() - t
print 'cd1 rbm_bernoulli %i_%i\tnumpy{%i}\t%.2f' %(
        nin, nout, batchsize, niter*batchsize/total_time)


