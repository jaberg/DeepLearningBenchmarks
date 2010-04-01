#!/bin/sh

# LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_ipp:$LD_LIBRARY_PATH ./mnist_example_ipp.x /data/lisa/data/mnist
# LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_noipp:$LD_LIBRARY_PATH ./mnist_example_noipp.x /data/lisa/data/mnist

LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_ipp:$LD_LIBRARY_PATH ./convnet_ipp.x > ${HOSTNAME}_eblearn_convnet_ipp.bmark
LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_ipp:$LD_LIBRARY_PATH ./convnet96_ipp.x > ${HOSTNAME}_eblearn_convnet96_ipp.bmark
LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_ipp:$LD_LIBRARY_PATH ./convnet256_ipp.x > ${HOSTNAME}_eblearn_convnet256_ipp.bmark

LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_noipp:$LD_LIBRARY_PATH ./convnet_noipp.x > ${HOSTNAME}_eblearn_convnet.bmark
LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_noipp:$LD_LIBRARY_PATH ./convnet96_noipp.x > ${HOSTNAME}_eblearn_convnet96.bmark
LD_LIBRARY_PATH=$PUB_PREFIX/eblearn_noipp:$LD_LIBRARY_PATH ./convnet256_noipp.x > ${HOSTNAME}_eblearn_convnet256.bmark
