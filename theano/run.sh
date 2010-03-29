#!/bin/bash

THEANO_FLAGS=device=cpu,floatX=float64 python mlp.py
THEANO_FLAGS=device=cpu,floatX=float32 python mlp.py
THEANO_FLAGS=device=gpu0,floatX=float32 python mlp.py
