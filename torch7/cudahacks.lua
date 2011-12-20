torch.CudaTensor.lab = {}

local lab = torch.CudaTensor.lab

function lab.randn(...)
   local t = torch.FloatTensor.lab.randn(...)
   return torch.Tensor(t:size()):copy(t)
end

-- local nn = torch.CudaTensor.nn

-- function nn.LogSoftMax_forward(self, input)
--    local t = torch.FloatTensor(input:size()):copy(input)
--    self.output = torch.FloatTensor()
--    return torch.FloatTensor.nn.LogSoftMax_forward(self, t)
-- end

