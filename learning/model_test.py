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

s = DP.create_random_sparse_tensor(len, size, in_channel, dimension=3.5)

net1 = ReNet1(in_channel, out_channel, dimension, alpha)
net2 = ReNet2(in_channel, out_channel, dimension, alpha)

#_, a1 = net1(s, check=True)
#_, a2 = net2(s, check=True)

