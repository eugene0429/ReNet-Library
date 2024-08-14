import torch
import MinkowskiEngine as ME
import numpy as np
import torch.nn as nn
import MinkowskiEngine.MinkowskiFunctional as MF
from utils.data_processing import data_processing as DP
from learning.model import EncoderBox, DecoderBox, Bridge, FinalDecoderBox, ConvBlock, DownConv, UpConv, FinalConv, PruningLayer, Net

class ConvBlock_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock_, self).__init__(D=3)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=3)
        self.norm = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x, size):
        x, _, _ = x.dense()
        x1 = DP.dense_to_sparse(x[..., 0], 3)
        x2 = DP.dense_to_sparse(x[..., 1], 3)
        x1 = self.conv(x1)
        x1 = DP.ensure_tensor_size(x1, size)
        x2 = self.conv(x2)
        x2 = DP.ensure_tensor_size(x2, size)
        x1, _, _ = x1.dense()
        x2, _, _ = x2.dense()
        x = torch.stack([x1, x2], dim=-1)
        x = DP.dense_to_sparse(x, 4)
        x = self.norm(x)
        x = self.relu(x)
        return x

len = 10
len_half = len//2
size = 64
in_channel = 1
out_channel = 1
dimension = 2

coords1 = torch.randint(0, size, (len, dimension))
coords2 = torch.randint(0, size, (len, dimension))

feats1 = torch.rand(len, in_channel)
feats2 = torch.rand(len, in_channel)

coords1, feats1 = ME.utils.sparse_collate([coords1], [feats1])
coords2, feats2 = ME.utils.sparse_collate([coords2], [feats2])


# new_col = torch.cat([
#     torch.zeros(len_half, 1, dtype=torch.int),
#     torch.ones(len_half, 1, dtype=torch.int)
# ], dim=0)

s1 = ME.SparseTensor(features=feats1, coordinates=coords1)
s2 = ME.SparseTensor(features=feats2, coordinates=coords2)

unet = Net(in_channel, out_channel, dimension)

_, a = unet(s1)
print(a)
#print(DP.sparse_to_dense_with_size(a, size))
