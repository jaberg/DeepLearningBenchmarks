import time
from theano.tensor import lscalar, lvector, matrix, tanh, dot, grad, log, arange
from theano.tensor.nnet import softmax
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
inputs=784
outputs=10
lr=numpy.asarray(0.01, dtype=config.floatX)

batchsize=60

data_x = shared(randn(n_examples, inputs))
data_y = shared(randint((n_examples,), outputs))

si = lscalar()
nsi = lscalar()
sx = data_x[si:si+nsi]
sy = data_y[si:si+nsi]

bmark = open("mlp_%s_%s.bmark" %(config.device, config.floatX), 'w')

def reportmodel(model, batchsize, t):
    bmark.write("%s\t" % model)
    if config.floatX == 'float32':
        prec = 'float'
    else:
        prec = 'double'
    bmark.write("theano{%s/%s/%i}\t" % (
        config.device[0], prec, batchsize))
    bmark.write("%.2f\n"%(n_examples/t)) # report examples / second

def eval_and_report(train, name):
    t = time.time()
    for i in xrange(n_examples):
        cost = train(i, 1)
        if not (i % 1000):
            print i, cost
    reportmodel(name, 1, time.time()-t)

    # repeat w batchsize
    t = time.time()
    for i in xrange(n_examples/batchsize):
        cost = train(i*batchsize, batchsize)
        if not (i % 20):
            print i*batchsize, cost
    reportmodel(name, batchsize, time.time()-t)

def bench_logreg():
    v = shared(zeros(outputs, inputs))
    c = shared(zeros(outputs))
    #
    # Note on the transposed-ness of v for some reason, this data layout is faster than the
    # non-transposed orientation.
    # The change doesn't make much difference in the deeper models, 
    # but in this case it was more than twice as fast.
    #

    p_y_given_x = softmax(dot(sx, v.T)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gv,gc = grad(cost, [v,c])

    train = function([si, nsi], [],
            updates={ v:v-lr*gv, c:c-lr*gc })

    eval_and_report(train, "mlp_784_10")

def bench_mlp_500():
    HUs=500
    w = shared(rand(HUs, inputs) * numpy.sqrt(6 / (inputs + HUs)))
    b = shared(zeros(HUs))
    v = shared(zeros(outputs,HUs))
    c = shared(zeros(outputs))

    p_y_given_x = softmax(dot(tanh(dot(sx, w.T)+b), v.T)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gw,gb,gv,gc = grad(cost, [w,b,v,c])

    train = function([si, nsi], cost,
            updates={ w:w-lr*gw,
                      b:b-lr*gb,
                      v:v-lr*gv,
                      c:c-lr*gc })
    eval_and_report(train, "mlp_784_500_10")

def bench_deep1000():
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

    p_y_given_x = softmax(dot(h2, v)+c)
    nll = -log(p_y_given_x)[arange(sy.shape[0]), sy]
    cost = nll.mean()

    gparams = grad(cost, params)

    train = function([si, nsi], cost,
            updates=[(p,p-lr*gp) for p,gp in zip(params, gparams)])
    eval_and_report(train, "mlp_784_1000_1000_1000_10")

if __name__ == '__main__':
    bench_logreg()
    bench_mlp_500()
    bench_deep1000()
