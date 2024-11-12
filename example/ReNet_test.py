import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from EnvioX.SparseTensorProcessor import SparseTensorProcessor # type: ignore
from ReNet.model1 import ReNet1 # type: ignore
from ReNet.model2 import ReNet2 # type: ignore

SP = SparseTensorProcessor()

len = 200
len_half = len//2
size = 64
in_channel = 3
out_channel = 3
dimension = 4
alpha = 0.5

s = SP.create_random_sparse_tensor(len,
                                   size,
                                   in_channel,
                                   dimension=3.5
                                   )

net = ReNet1(in_channel,
             out_channel,
             dimension,
             alpha
             )

_, a1 = net(s, check=True)

