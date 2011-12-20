local SpatialConvolutionFast, parent = torch.class('nn.SpatialConvolutionFast', 'nn.Module')

function SpatialConvolutionFast:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self:reset()
end

function SpatialConvolutionFast:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function SpatialConvolutionFast:forward(input)   
   input = input:unfold(2, self.kH, self.dH)
   input = input:unfold(3, self.kW, self.dW)
   input = input:transpose(2,4)
   input = input:transpose(3,5)

   self.finput:resize(self.kW*self.kH*self.nInputPlane, input:size(4)*input:size(5)):copy(input)

   self.output:resize(self.nOutputPlane, input:size(4), input:size(5))
   local output = input.new(self.output:storage(), 1, self.nOutputPlane, -1, input:size(4)*input:size(5), -1):copy(
      input.new(self.bias:storage(), 1, self.nOutputPlane, 1, input:size(4)*input:size(5), 0))

   output:addmm(1, self.weight, self.finput)
   return self.output
end

function SpatialConvolutionFast:backward(input, gradOutput)
   if self.gradInput then
      gradOutput = input.new(gradOutput:storage(), 1, gradOutput:size(1), -1, gradOutput:size(2)*gradOutput:size(3), -1)
      
      self.fgradInput:resizeAs(self.finput):zero()
      self.fgradInput:addmm(1, self.weight:t(), gradOutput)
      
      self.gradInput:resizeAs(input):zero()
      local gradInput = self.gradInput:unfold(2, self.kH, self.dH)
      gradInput = gradInput:unfold(3, self.kW, self.dW)
      gradInput = gradInput:transpose(2,4)
      gradInput = gradInput:transpose(3,5)
      gradInput:add(self.fgradInput)

      return self.gradInput

   end
end

function SpatialConvolutionFast:accGradParameters(input, gradOutput, scale)
   gradOutput = input.new(gradOutput:storage(), 1, gradOutput:size(1), -1, gradOutput:size(2)*gradOutput:size(3), -1)
   self.gradWeight:addmm(1, gradOutput, self.finput:t())
   input.new(self.gradBias:storage(), 1, gradOutput:size(1), 1, gradOutput:size(2), 0):add(gradOutput)
end
