require "lab"

local MiniBatchGradient = torch.class('nn.MiniBatchGradient')

function MiniBatchGradient:__init(module, criterion, batchSize)
   self.learningRate = 0.01
   self.learningRateDecay = 0
   self.maxIteration = 25
   self.shuffleIndices = true
   self.module = module
   self.criterion = criterion
   self.batchSize = batchSize or -1
end

function MiniBatchGradient:train(dataset)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion

   local shuffledIndices = lab.randperm(dataset:size())
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   end

   -- fully batch?
   if self.batchSize < 1 then
      self.batchSize = dataset:size()
   end

   print("# MiniBatchGradient: training with batch size: " .. self.batchSize)

   while true do
      local currentError = 0
      module:zeroGradParameters()
      for t = 1,dataset:size() do
         local example = dataset[shuffledIndices[t]]
         local input = example[1]
         local target = example[2]

         currentError = currentError + criterion:forward(module:forward(input), target)

         module:backward(input, criterion:backward(module.output, target))

         if t % self.batchSize == 0 then
            module:updateParameters(currentLearningRate)
            module:zeroGradParameters()
         end

         if self.hookExample then
            self.hookExample(self, example)
         end
      end

      if self.hookIteration then
         self.hookIteration(self, iteration)
      end

      currentError = currentError / dataset:size()
      print("# current error = " .. currentError)
      iteration = iteration + 1
      currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
         print("# MiniBatchGradient: you have reached the maximum number of iterations")
         break
      end
   end
end

function MiniBatchGradient:write(file)
   file:writeDouble(self.learningRate)
   file:writeDouble(self.learningRateDecay)
   file:writeInt(self.maxIteration)
   file:writeBool(self.shuffleIndices)
   file:writeObject(self.module)
   file:writeObject(self.criterion)
   file:writeLong(self.batchSize)
end

function MiniBatchGradient:read(file)
   self.learningRate = file:readDouble()
   self.learningRateDecay = file:readDouble()
   self.maxIteration = file:readInt()
   self.shuffleIndices = file:readBool()
   self.module = file:readObject()
   self.criterion = file:readObject()
   self.batchSize = file:readLong()
end
