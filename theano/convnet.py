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

n_examples=6000
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

batchsize=60

data_32x32 = shared(randn(n_examples, 1, 32, 32))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_32x32[si:si+nsi]
sy = data_y[si:si+nsi]

bmark = open("%s_convnet_%s_%s.bmark"% (socket.gethostname(), config.device, config.floatX), 'w')

def reportmodel(model, batchsize, t):
    bmark.write("%s\t" % model)
    bmark.write("theano w batchsize=%i device=%s dtype=%s\t" % (
        batchsize, config.device, config.floatX))
    bmark.write("%.2f\n"%t)

def eval_and_report(train, name, batchsizes):
    for bs in batchsizes:
        assert n_examples % bs == 0 # can't be cheatin now...
        t = time.time()
        for i in xrange(n_examples/bs):
            cost = train(i*bs, bs)
            if not (i % (1000/bs)):
                print i*bs, cost
        reportmodel(name, bs, time.time()-t)

def bench_lenet5_like_32x32(batchsize):
    w0 = shared(rand(6, 1, 5, 5) * numpy.sqrt(6 / (25.)))
    b0 = shared(zeros(6))
    w1 = shared(rand(20, 6, 5, 5) * numpy.sqrt(6 / (25.)))
    b1 = shared(zeros(20))
    vv = shared(rand(20*5*5, 120) * numpy.sqrt(6.0/20./25))
    cc = shared(zeros(120))
    v = shared(zeros(120, outputs))
    c = shared(zeros(outputs))
    params = [w0, b0, w1, b1, v, c, vv, cc]

    c0 = tanh(conv2d(sx, w0, image_shape=(batchsize, 1, 32, 32), filter_shape=(6, 1, 5, 5)) + b0.dimshuffle(0, 'x', 'x'))
    s0 = tanh(max_pool2D(c0, (2,2))) # this is not the correct leNet5 model, but it's closer to
    # the real
    # model than the conv-with-1/4 that torch5 implements.

    c1 = tanh(conv2d(s0, w1, image_shape=(batchsize, 6, 14, 14), filter_shape=(20,6,5,5)) + b1.dimshuffle(0, 'x', 'x'))
    s1 = tanh(max_pool2D(c1, (2,2)))

    print s1.flatten(2).type, vv.type, cc.type, tanh(dot(s1.flatten(2), vv)+cc).type
    p_y_given_x = softmax(dot(tanh(dot(s1.flatten(2), vv)+cc), v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])

    eval_and_report(train, "convnet_32x32_c5x5_s2x2_c5x5_s2x2_120_10", [batchsize])

if __name__ == '__main__':
    bench_lenet5_like_32x32(1)
    bench_lenet5_like_32x32(50)

