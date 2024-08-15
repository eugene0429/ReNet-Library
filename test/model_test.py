import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from util.data_processing import DataProcess as DP
from learning.model1 import Net1
from learning.model2 import Net2

len = 10
len_half = len//2
size = 64
in_channel = 3
out_channel = 3
dimension = 4

s = DP.create_random_sparse_tensor_4D(len, size, in_channel)

net1 = Net1(in_channel, out_channel, dimension)
net2 = Net2(in_channel, out_channel, dimension)

_, a1 = net1(s)
_, a2 = net2(s)
