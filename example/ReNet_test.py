import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import EnvioX
import ReNet

len = 200
len_half = len//2
size = 64
in_channel = 3
out_channel = 3
dimension = 4
alpha = 0.5

s = EnvioX.SP.create_random_sparse_tensor(len,
                                          size,
                                          in_channel,
                                          dimension=3.5
                                          )

net = ReNet.ReNet1(in_channel,
                  out_channel,
                  dimension,
                  alpha
                  )

_, o = net(s, check=True)

