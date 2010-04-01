import time, socket
from theano.tensor import lscalar, lvector, matrix, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool2D
from theano import shared, function, config
import numpy, theano
from numpy import asarray, random
random.seed(2344)

def rand(*size):
    return asarray(random.rand(*size), dtype=config.floatX)
def randn(*size):
    return asarray(random.randn(*size), dtype=config.floatX)
def randint(size, high):
    return asarray(random.randint(size=size, low=0, high=high), dtype='int32')
def zeros(*size):
    return numpy.zeros(size, dtype=config.floatX)

n_examples=1000
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

data_x = shared(randn(n_examples, 1, 32, 32))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si+nsi]
sy = data_y[si:si+nsi]

bmark = open("%s_convnet_%s_%s.bmark"% (socket.gethostname(), config.device, config.floatX), 'w')

if config.floatX == 'float32':
    prec = 'float'
else:
    prec = 'double'

def reportmodel(model, batchsize, v):
    bmark.write("%s\t" % model)
    bmark.write("theano{%s/%s/%i}\t" % (
        config.device[0], prec, batchsize))
    bmark.write("%.2f\n"%v)

def eval_and_report(train, name, batchsizes, N=n_examples):
    for bs in batchsizes:
        assert N % bs == 0 # can't be cheatin now...
        t = time.time()
        for i in xrange(N/bs):
            cost = train(i*bs, bs)
            if not (i % (1000/bs)):
                print i*bs, cost
        reportmodel(name, bs, N/(time.time()-t))

def bench_ConvSmall(batchsize):
    data_x.value = randn(n_examples, 1, 32, 32)
    w0 = shared(rand(6, 1, 5, 5) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 5, 5) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*5*5, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 32, 32), filter_shape=(6, 1, 5, 5)) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool2D(c0, (2,2))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 14, 14), filter_shape=(16,6,5,5)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool2D(c1, (2,2)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])

    eval_and_report(train, "ConvSmall", [batchsize], N=600)

def bench_ConvMed(batchsize):
    data_x.value = randn(n_examples, 1, 96, 96)
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*8*8, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 96, 96), filter_shape=(6,1,7,7)) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool2D(c0, (3,3))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 30, 30), filter_shape=(16,6,7,7)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool2D(c1, (3,3)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    eval_and_report(train, "ConvMed", [batchsize], N=120)

def bench_ConvLarge(batchsize):
    data_x.value = randn(n_examples, 1, 256, 256)
    w0 = shared(rand(6, 1, 7, 7) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(16, 6, 7, 7) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(16))
    vv = shared(rand(16*11*11, 120) * numpy.sqrt(6.0/16./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 256, 256), filter_shape=(6,1,7,7)) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool2D(c0, (5,5))) # this is not the correct leNet5 model, but it's closer to

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 50, 50), filter_shape=(16,6,7,7)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool2D(c1, (4,4)))

    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    eval_and_report(train, "ConvLarge", [batchsize], N=120)

if __name__ == '__main__':
    bench_ConvSmall(1)
    bench_ConvSmall(60)
    bench_ConvMed(1)
    bench_ConvMed(60)
    bench_ConvLarge(1)
    bench_ConvLarge(60)

