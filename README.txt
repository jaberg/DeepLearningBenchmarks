Intro
=====


The benchmarking folder contains efforts to benchmark Theano against various
other systems.  Each subfolder corresponds to a particular style of
implementation.  Since there is a variety of benchmark problems and of software
systems, there isn't a standard for how to run the benchmark suite.  There is
however a standard for how each benchmark should produce results.  Every
benchmark run should produce one or more files with the results of benchmarking.
These files must end with extension '.bmark'.  These files must have 'csv' lines
like this:

task<tab>implementation name<tab>examples/second


Current Tasks
==============

Dense
-----

mlp_784_10
- training on 10K MNIST-sized examples with unregularized Logistic Regression (crossentropy / NLL error)

mlp_784_500_10
- training on 10K examples with a single-hidden layer model with 500 hidden units

mlp_784_1000_1000_1000_10
- training on 10K examples with multiple hidden layers 

cd1 rbm_bernoulli 1024_1024
- train an RBM from 10K 1024-dimensional inputs

daa_1024_1024
- train a denoising autoassociator from 10K 1024-dimensional inputs

Convolutional
-------------

LeNet5_32x32
- train from 10K 32x32 inputs, as in LeNet5

LeNet7_96x96
- train from 10K 96x96 images

LeNet7_256x256x3
- train from 10K 256x256 rgb images








Potential Tasks
================

Dense
-----

mlp_32_10
- training on 10K tiny examples with unregularized Logistic Regression (crossentropy / NLL error)

mlp_784_10 with L1
- training on 10K examples with L1 regularization

mlp_784_10 with L2
- training on 10K examples with L2 regularization

aa_64_64
- train an autoassociator from at 10K 64-dimensional inputs

aa_1024_1024
- train an autoassociator from 10K 1024-dimensional inputs


cd1 rbm_bernoulli 64_64
- train an RBM from 10K 64-dimensional inputs



Convolutional
-------------


LeNet5_32x32x3
- train from 10K Tiny-Image sized inputs (in color)

conv_daa_i32x32_f7x7
- train a convolutional

conv_daa_i256x256_f9x9
- train a convolutional


Recurrent
---------


