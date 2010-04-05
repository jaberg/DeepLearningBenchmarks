#!/bin/sh

./mlp.lua

mv torch5.bmark ${HOSTNAME}_torch5.bmark

./mlp_minibatch.lua
mv torch5_minibatch.bmark ${HOSTNAME}_torch5_minibatch.bmark
