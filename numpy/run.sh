#!/bin/sh

python mlp.py 784 500 10 1 1000 > ${HOSTNAME}_mlp_1.bmark
python mlp.py 784 500 10 60 100 > ${HOSTNAME}_mlp_60.bmark

python logreg.py 784 10 1 1000 > ${HOSTNAME}_lr_784_1.bmark
python logreg.py 784 10 60 100 > ${HOSTNAME}_lr_784_60.bmark
python logreg.py 32  10 1 1000 > ${HOSTNAME}_lr_32_1.bmark
python logreg.py 32  10 60 100 > ${HOSTNAME}_lr_32_60.bmark
