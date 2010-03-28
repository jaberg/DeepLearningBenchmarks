Intro
=====


The benchmarking folder contains efforts to benchmark Theano against various
other systems.  Each subfolder corresponds to a particular type of computation,
and each sub-subfolder corresponds to the implementation of that computation in
with a particular software package.

Since there is a variety of benchmark problems and of software systems, there
isn't a standard for how to run the benchmark suite.
There is however a standard for how each benchmark should produce results.
Every benchmark run should produce one or more files with the results of
benchmarking. These files must end with extension '.bmark'.  These files must
have at least three lines each:

1) line 1 - description of computation/problem
2) line 2 - description of implementation/platform
3) line 3 - time required (in seconds)
4) line 4 - [optional] an estimated number of FLOPS performed (not necessarily same for all implementations of problem)




Tasks
======

Dense
-----

mlp_32_10_xent
- training on 50000 tiny examples with unregularized Logistic Regression (crossentropy / NLL error)

mlp_784_10_xent
- training on 50000 MNIST-sized examples with unregularized Logistic Regression (crossentropy / NLL error)

mlp_784_10_xent with L1
- training on 50000 examples with L1 regularization

mlp_784_10_xent with L2
- training on 50000 examples with L2 regularization

mlp_784_500_10_xent
- training on 50000 examples with a single-hidden layer model with 500 hidden units

mlp_784_1000_1000_1000_10_xent
- training on 50000 examples with multiple hidden layers 

mlp_784_1000_1000_1000_10_xent
- training on 50000 examples with multiple hidden layers 

aa_64_64
- train an autoassociator from at 50K 64-dimensional inputs

aa_1024_1024
- train an autoassociator from 50K 1024-dimensional inputs

daa_1024_1024
- train a denoising autoassociator from 50K 1024-dimensional inputs

cd-1 rbm_bernoulli 64_64
- train an RBM from 50K 64-dimensional inputs

cd-1 rbm_bernoulli 1024_1024
- train an RBM from 50K 1024-dimensional inputs


Convolutional
-------------

LeNet5_32x32
- train from 50K 32x32 inputs, as in LeNet5

LeNet5_32x32x3
- train from 50K Tiny-Image sized inputs (in color)

LeNet7_96x96
- train from 50K 96x96 images

LeNet7_256x256
- train from 50K 256x256 images

conv_daa_i32x32_f7x7
- train a convolutional

conv_daa_i256x256_f9x9
- train a convolutional


Recurrent
---------


