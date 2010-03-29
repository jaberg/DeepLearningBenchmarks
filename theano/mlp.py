import time
from theano.tensor import matrix, tanh, dot, grad
from theano import shared, function, config
import numpy
from numpy import asarray, random
random.seed(2344)

def rand(*size):
    return asarray(random.rand(*size), dtype=config.floatX)
def randn(*size):
    return asarray(random.randn(*size), dtype=config.floatX)
def zeros(*size):
    return numpy.zeros(size, dtype=config.floatX)

n_examples=10000
inputs=784
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

data_x = randn(n_examples, inputs)
data_y = randn(n_examples, outputs)

bmark = open("mlp.bmark", 'w')

def bench_logreg():
    if True: # MLP 784/10
        sx = matrix()
        sy = matrix()
        v = shared(zeros(inputs, outputs))
        c = shared(zeros(outputs))

        cost = (((dot(sx, v)+c) - sy)**2).mean()

        gv,gc = grad(cost, [v,c])

        train = function([sx, sy], [],
                updates={ v:v-lr*gv, c:c-lr*gc })

        t = time.time()
        for i in xrange(n_examples):
            cost = train(data_x[i:i+1], data_y[i:i+1])
            if not (i % 1000):
                print i, cost
        bmark.write("mlp_784_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("%.2f\n"%(time.time()-t))

        # repeat w batchsize 50
        t = time.time()
        for i in xrange(n_examples/50):
            cost = train(data_x[i*50:(i+1)*50], data_y[i*50:(i+1)*50])
            if not (i % 20):
                print i, cost
        bmark.write("mlp_784_10\t")
        bmark.write("theano w batchsize=50\t")
        bmark.write("%.2f\n"%(time.time()-t))
    else:
        bmark.write("# mlp_784_500_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("0\n")

def bench_mlp_500():
    if True: # MLP 784/500/10, batchsize=1

        HUs=500
        sx = matrix()
        sy = matrix()
        w = shared(rand(inputs, HUs) * numpy.sqrt(6 / (inputs + HUs)))
        b = shared(zeros(HUs))
        v = shared(zeros(HUs, outputs))
        c = shared(zeros(outputs))

        cost = (((dot(tanh(dot(sx, w)+b), v)+c) - sy)**2).sum()

        gw,gb,gv,gc = grad(cost, [w,b,v,c])

        train = function([sx, sy], cost,
                updates={ w:w-lr*gw,
                          b:b-lr*gb,
                          v:v-lr*gv,
                          c:c-lr*gc })

        t = time.time()
        for i in xrange(n_examples):
            cost = train(data_x[i:i+1], data_y[i:i+1])
            if not (i % 1000):
                print i, cost
        bmark.write("mlp_784_500_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("%.2f\n"%(time.time()-t))
    else:
        bmark.write("# mlp_784_500_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("0\n")

def bench_deep_mlp_500():
    if True: # MLP 784/500/10, batchsize=1

        sx = matrix()
        sy = matrix()
        w0 = shared(rand(inputs, 1000) * numpy.sqrt(6 / (inputs + 1000)))
        b0 = shared(zeros(1000))
        w1 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
        b1 = shared(zeros(1000))
        w2 = shared(rand(1000, 1000) * numpy.sqrt(6 / (1000+1000)))
        b2 = shared(zeros(1000))
        v = shared(zeros(1000, outputs))
        c = shared(zeros(outputs))
        params=[w0,b0,w1,b1,w2,b2,v,c]

        h0 = tanh(dot(sx, w0)+b0)
        h1 = tanh(dot(h0, w1)+b1)
        h2 = tanh(dot(h1, w2)+b2)

        cost = (((dot(h2, v)+c) - sy)**2).sum()

        gparams = grad(cost, params)

        train = function([sx, sy], cost,
                updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])

        t = time.time()
        for i in xrange(n_examples):
            cost = train(data_x[i:i+1], data_y[i:i+1])
            if not (i % 1000):
                print i, cost
        bmark.write("mlp_784_1000_1000_1000_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("%.2f\n"%(time.time()-t))
    else:
        bmark.write("# mlp_784_1000_1000_1000_10\t")
        bmark.write("theano w batchsize=1\t")
        bmark.write("0\n")
