import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from util.data_processing import DataProcessing
from model1 import ReNet1
from model2 import ReNet2

DP = DataProcessing()

len = 200
len_half = len//2
size = 64
in_channel = 3
out_channel = 3
dimension = 4
alpha = 0.5

# s = DP.create_random_sparse_tensor(len, size, in_channel, dimension=3.5)
# net = ReNet1(in_channel, out_channel, dimension, alpha)
# _, a1 = net(s, check=True)

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import torch

conv = ME.MinkowskiConvolution(3, 1, kernel_size=1, stride=1, dimension=3)
down_conv = ME.MinkowskiConvolution(3, 3, kernel_size=3, stride=2, dimension=3)
up_conv = ME.MinkowskiGenerativeConvolutionTranspose(3, 3, kernel_size=2, stride=2, dimension=3)

s = DP.create_random_sparse_tensor(4, 4, 3, 3)
print(s)
s = down_conv(s)
print(s)
s = DP.generate_empty_sparse_tensor(s)
print(s)
s = up_conv(s)
s = MF.sigmoid(conv(s))
print(s)
