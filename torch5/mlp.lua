#!/usr/bin/env lua

require "lab"
require "os"
require "nn"


n_examples=6000;
outputs=10;

io.output("torch5.bmark")


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
    trainer = nn.StochasticGradient(mlp, criterion)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write(string.format("mlp_%i_%i", inputs, outputs), "\t",
        "torch5", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
else
    io.write(string.format("# mlp_%i_%i", inputs, outputs), "\t",
        "torch5", "\t",
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
    trainer = nn.StochasticGradient(mlp, criterion)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write(string.format("mlp_%i_500_%i", inputs, outputs), "\t",
        "torch5", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
else
    io.write(string.format("# mlp_%i_500_%i", inputs, outputs), "\t",
        "torch5", "\t",
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
    trainer = nn.StochasticGradient(mlp, criterion)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dataset)
    -- we're not using Xent, but using Xent would be even slower
    io.write("mlp_784_1000_1000_1000_10", "\t",
        "torch5", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")

else
    io.write("# mlp_784_1000_1000_1000_10", "\t",
        "torch5", "\t",
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
    trainer = nn.StochasticGradient(mlp, criterion)

    trainer.learningRate = 0.01
    trainer.shuffleIndices = false
    trainer.maxIteration = 1
    local x = os.clock()
    trainer:train(dset_32x32)
    -- we're not using Xent, but using Xent would be even slower
    io.write("convnet_32x32_c5x5_s2x2_c5x5_s2x2_120_10", "\t",
        "torch5", "\t",
        string.format("%.2f\n", n_examples/(os.clock() - x)), "\n")
end
