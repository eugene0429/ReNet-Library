import torch
import MinkowskiEngine as ME
import numpy as np
import torch.nn as nn
import MinkowskiEngine.MinkowskiFunctional as MF
from data_processing import data_processing as DP
from model import EncoderBox, DecoderBox, Bridge, FinalDecoderBox, ConvBlock, DownConv, UpConv, FinalConv, PruningLayer, Net

class ConvBlock1(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1, self).__init__(D=3)
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

class UpConv_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(UpConv_, self).__init__(D)
        self.up_conv = ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.norm(x)
        x = MF.elu(x)
        return x

class Bridge_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(Bridge_, self).__init__(D)
        self.conv = ConvBlock(in_channels, out_channels, D)
        self.up_conv = UpConv_(out_channels, out_channels, D)

    def forward(self, x):
        x = self.conv(x)
        x = self.up_conv(x)
        return x

class DecoderBox_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, alpha):
        super(DecoderBox_, self).__init__(D)
        self.conv = ConvBlock(in_channels, out_channels, D)
        self.up_conv = UpConv_(out_channels, out_channels, D)

    def forward(self, x, skip_connection):
        x = ME.cat(x, skip_connection)
        x = self.conv(x)
        x = self.up_conv(x)
        return x

class FinalDecoderBox_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, mid_channels, out_channels, D, alpha):
        super(FinalDecoderBox_, self).__init__(D)
        self.conv1 = ConvBlock(in_channels, mid_channels, D)
        self.pruning = PruningLayer(mid_channels, D, alpha)
        self.conv2 = FinalConv(mid_channels, out_channels, D)

    def forward(self, x, skip_connection):
        x = ME.cat(x, skip_connection)
        x = self.conv1(x)
        lh, x = self.pruning(x)
        x = self.conv2(x)
        return lh, x

class net(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, alpha=0.5):
        super(net, self).__init__(D)
        self.enc1 = EncoderBox(in_channels, in_channels, D)
        self.enc2 = EncoderBox(in_channels, in_channels, D)
        self.bridge = Bridge(in_channels, in_channels, D)
        self.dec2 = DecoderBox(2*in_channels, in_channels, D, alpha, 2)
        self.dec1 = FinalDecoderBox(2*in_channels, in_channels, out_channels, D, alpha, 1)
    def forward(self, x):
        print(DP.sparse_to_dense_with_size(x, 8))
        skip1, x = self.enc1(x) # [1,1,1], [2,2,2]
        print(DP.sparse_to_dense_with_size(x, 8))
        skip2, x = self.enc2(x) # [2,2,2], [4,4,4]
        print(DP.sparse_to_dense_with_size(x, 8))
        x = self.bridge(x) # [2,2,2]
        print(DP.sparse_to_dense_with_size(x, 8))
        lh1, x = self.dec2(x, skip2) # [1,1,1]
        print(DP.sparse_to_dense_with_size(x, 8))
        lh2, x = self.dec1(x, skip1) # [1,1,1]
        print(DP.sparse_to_dense_with_size(x, 8))
        return x
    
class net_(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, alpha=0.5):
        super(net_, self).__init__(D)
        self.enc1 = EncoderBox(in_channels, in_channels, D)
        self.enc2 = EncoderBox(in_channels, in_channels, D)
        self.bridge = Bridge_(in_channels, in_channels, D)
        self.dec2 = DecoderBox_(2*in_channels, in_channels, D, alpha)
        self.dec1 = FinalDecoderBox_(2*in_channels, in_channels, out_channels, D, alpha)
    def forward(self, x):
        print(DP.sparse_to_dense_with_size(x, 8))
        skip1, x = self.enc1(x)
        print(DP.sparse_to_dense_with_size(x, 8))
        skip2, x = self.enc2(x)
        print(DP.sparse_to_dense_with_size(x, 8))
        x = self.bridge(x)
        print(DP.sparse_to_dense_with_size(x, 8))
        x = self.dec2(x, skip2)
        print(DP.sparse_to_dense_with_size(x, 8))
        lh2, x = self.dec1(x, skip1)
        print(DP.sparse_to_dense_with_size(x, 8))
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
net1 = net(in_channel, out_channel, dimension)
net2 = net_(in_channel, out_channel, dimension)

_, a = unet(s1)
print(a)
#print(DP.sparse_to_dense_with_size(a, size))
