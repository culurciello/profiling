-- E. Culurciello, Teradeep Inc.
--
-- filter-bank industrystandard deep learning profiling test:
-- Multiple filtering operations in full CNN manner (Conv, Maxp, Threshold)

require 'nnx'
torch.setdefaulttensortype('torch.FloatTensor')

local nk0 = 4 -- input planes / maps
local nk1 = 16 -- output planes / maps
local is1 = 9 -- convolutional filter size
local ss1 = 2 -- pooling size
local imSize = 500 -- input image size

-- network to be profiled:
local network = nn.Sequential()
network:add(nn.SpatialConvolution(nk0, nk1, is1, is1))
network:add(nn.SpatialMaxPooling(ss1,ss1,ss1,ss1))
network:add(nn.Threshold())

-- total number of operations:
local opmac = 2 -- ops per MAC
local c1_ops = opmac * nk0*nk1*is1^2*(imSize-is1+1)^2 -- convolution operations
local p1_ops = nk1*ss1^2*((imSize-is1+1)-ss1+1)^2 -- pooling operations (general, not just max)
local n1_ops = opmac * nk1*(((imSize-is1+1)-ss1+1)/2)^2 -- nonlinear operations, 1 segment linear approximations
local t_ops = c1_ops + p1_ops + n1_ops -- total network operations 
print('Number of operations in test network: [G]', t_ops/1e9)

-- input tensor:
local src = torch.Tensor(nk0, imSize, imSize)

-- profiling timers:
timer = torch.Timer()

-- profile multiple times:
for i=1,5 do
   timer:reset()
   network:forward(src)
   t_hw = timer:time().real
   print('Processing time [ms]:', t_hw*1000, 'G-ops/s:', t_ops/t_hw/1e9)
end
print('Measure device + memory power while processing to measure G-ops/s/Watt')
