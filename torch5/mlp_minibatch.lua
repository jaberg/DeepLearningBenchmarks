#!/usr/bin/env lua

require "lab"
require "os"
require "nn"

-- When discussin Torch's performance by email,
-- Ronan sent me this file for doing mini-batches.  
-- It seems like an unofficial feature, I don't know why this isn't in the
-- main distribution...
dofile('MiniBatchGradient.lua')


n_examples=12000;
outputs=10;

io.output("torch5_minibatch.bmark")

if true then -- MLP 32/10
    dataset={};
    function dataset:size() return n_examples end
    inputs=32;
    for i=1,dataset:size() do 
      dataset[i] = {lab.randn(inputs), (i % outputs)+1}
    end
    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.Linear(inputs, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write(string.format("mlp_%i_%i", inputs, outputs), "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
end


dataset={};
function dataset:size() return n_examples end
inputs=784;


for i=1,dataset:size() do 
  dataset[i] = {lab.randn(inputs), (i % outputs)+1}
end

if true -- MLP 784/10
then
    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.Linear(inputs, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write(string.format("mlp_%i_%i", inputs, outputs), "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
else
    io.write(string.format("# mlp_%i_%i", inputs, outputs), "\t",
        "torch5{60}", "\t",
        "0.0", "\n")
end


if true -- MLP 784/500/10
then

    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.Linear(inputs, 500))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(500, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write(string.format("mlp_%i_500_%i", inputs, outputs), "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
else
    io.write(string.format("# mlp_%i_500_%i", inputs, outputs), "\t",
        "torch5{60}", "\t",
        "0.0", "\n")
end


if true --MLP 784/1000/1000/1000/10
then

    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.Linear(inputs, 1000))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(1000, 1000))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(1000, 1000))
    mlp:add(nn.Tanh())
    mlp:add(nn.Linear(1000, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write("mlp_784_1000_1000_1000_10", "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")

else
    io.write("# mlp_784_1000_1000_1000_10", "\t",
        "torch5{60}", "\t",
        "0.0", "\n")

end

dset_32x32={};
function dset_32x32:size() return n_examples end
for i=1,dset_32x32:size() do 
  dset_32x32[i] = {lab.randn(32,32,1), (i % outputs)+1}
end

if true --LeNet5-like 32x32
then

    -- There is no max-pooling implemented, just avg pooling.
    -- So I added tanh between every layer to separate true conv layers with
    -- the subsampling (which is just a convolution with 1s)

    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.SpatialConvolution(1, 6, 5, 5)) -- output 28x28
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(6, 2, 2, 2, 2)) --output 14x14
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialConvolution(6, 16, 5, 5)) -- output 10x10
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(16, 2, 2, 2, 2)) -- output 5x5
    mlp:add(nn.Tanh())
    mlp:add(nn.Reshape(16*5*5))
    mlp:add(nn.Linear(16*5*5, 120))
    mlp:add(nn.Linear(120, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dset_32x32)
    -- we're not using Xent, but using Xent would be even slower
    io.write("ConvSmall", "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
end

dset_96x96={};
function dset_96x96:size() return 100 end
for i=1,dset_96x96:size() do 
  dset_96x96[i] = {lab.randn(96,96,1), (i % outputs)+1}
end

if true --LeNet5-like 96x96
then

    -- There is no max-pooling implemented, just avg pooling.
    -- So I added tanh between every layer to separate true conv layers with
    -- the subsampling (which is just a convolution with 1s)

    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.SpatialConvolution(1, 6, 7, 7)) -- output 90x90
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(6, 3, 3, 3, 3)) --output 30x30
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialConvolution(6, 16, 7, 7)) -- output 24x24
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(16, 3, 3, 3, 3)) -- output 8x8
    mlp:add(nn.Tanh())
    mlp:add(nn.Reshape(16*8*8))
    mlp:add(nn.Linear(16*8*8, 120))
    mlp:add(nn.Linear(120, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dset_96x96)
    -- we're not using Xent, but using Xent would be even slower
    io.write("ConvMed", "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", dset_96x96:size()/(os.clock() - x)), "\n")
end


dset_256x256={};
function dset_256x256:size() return 20 end
for i=1,dset_256x256:size() do 
  dset_256x256[i] = {lab.randn(256,256,1), (i % outputs)+1}
end

if true --LeNet5-like 256x256
then

    -- There is no max-pooling implemented, just avg pooling.
    -- So I added tanh between every layer to separate true conv layers with
    -- the subsampling (which is just a convolution with 1s)

    mlp = nn.Sequential();                 -- make a multi-layer perceptron
    mlp:add(nn.SpatialConvolution(1, 6, 7, 7)) -- output 250x250
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(6, 5, 5, 5, 5)) --output 50x50
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialConvolution(6, 16, 7, 7)) -- output 44x44
    mlp:add(nn.Tanh())
    mlp:add(nn.SpatialSubSampling(16, 4, 4, 4, 4)) -- output 11x11
    mlp:add(nn.Tanh())
    mlp:add(nn.Reshape(16*11*11))
    mlp:add(nn.Linear(16*11*11, 120))
    mlp:add(nn.Linear(120, outputs))
    mlp:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()  
    trainer = nn.MiniBatchGradient(mlp, criterion, 60)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dset_256x256)
    -- we're not using Xent, but using Xent would be even slower
    io.write("ConvLarge", "\t",
        "torch5{60}", "\t",
        string.format("%.2f\n", dset_256x256:size()/(os.clock() - x)), "\n")
end
