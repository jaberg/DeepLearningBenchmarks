import theano
from theano.misc.check_blas import execute

sizes = [500, 1000, 1500, 2000, 2500]
iters = 10

for order in ['c', 'f']:
    for size in sizes:
        t = execute(verbose=False, M=size, N=size, K=size, iters=iters)[0]
        print "gemm theano{order_%s/%s/%d/%d}" % (
            order, theano.config.floatX, iters, size), t
