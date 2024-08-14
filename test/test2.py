import torch
import torch.nn as nn
import torch.nn.functional as F

class PruningLayer(nn.Module):
    def __init__(self, in_channels, alpha=0.5):
        super(PruningLayer, self).__init__()
        self.alpha = alpha
        self.likelihood_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        likelihood_map = torch.sigmoid(self.likelihood_conv(x))
        mask = (likelihood_map >= self.alpha).float()
        pruned_features = x * mask  # Element-wise multiplication to zero-out low likelihood features
        return likelihood_map, pruned_features

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.elu(x)
        return x

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.elu(x)
        return x

class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        
    def forward(self, x):
        x = self.up_conv(x)
        x = F.elu(x)
        return x

class EncoderBox(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBox, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.down_conv = DownConv(out_channels, out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x_down = self.down_conv(x)
        return x, x_down

class DecoderBox(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super(DecoderBox, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pruning = PruningLayer(out_channels, alpha)
        self.up_conv = UpConv(out_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        lh, x = self.pruning(x)
        x = self.up_conv(x)
        return lh, x

class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.up_conv = UpConv(out_channels, out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.up_conv(x)
        return x

class FinalDecoderBox(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, alpha):
        super(FinalDecoderBox, self).__init__()
        self.conv1 = ConvBlock(in_channels, mid_channels)
        self.pruning = PruningLayer(mid_channels, alpha)
        self.conv2 = FinalConv(mid_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv1(x)
        lh, x = self.pruning(x)
        x = self.conv2(x)
        return lh, x

class Net(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.5):
        super(Net, self).__init__()
        self.enc1 = EncoderBox(in_channels, in_channels)
        self.enc2 = EncoderBox(in_channels, in_channels)
        self.bridge = Bridge(in_channels, in_channels)
        self.dec2 = DecoderBox(2*in_channels, in_channels, alpha)
        self.dec1 = FinalDecoderBox(2*in_channels, in_channels, out_channels, alpha)
        
    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        x = self.bridge(x)
        lh1, x = self.dec2(x, skip2)
        lh2, x = self.dec1(x, skip1)
        return x

import MinkowskiEngine as ME
from utils.data_processing import data_processing as DP

size = 8
len = 4

a = torch.zeros(size, size)
idx_x = torch.randint(0, size, (len,))
idx_y = torch.randint(0, size, (len,))

coords = torch.stack([idx_x, idx_y], dim=0).T
feats = torch.rand(len,1)

coords, feats = ME.utils.sparse_collate([coords], [feats])
s = ME.SparseTensor(features=feats, coordinates=coords)

a[idx_x, idx_y] = torch.rand(len,)
a = a.unsqueeze(0)
a = a.unsqueeze(0)

net = Net(1, 1)

conv1_1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
conv1_2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
conv2 = ME.MinkowskiConvolution(1, 1, kernel_size=3, stride=2, dimension=2)
upconv1 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, padding=1, bias=False)
upconv2 = ME.MinkowskiGenerativeConvolutionTranspose(1, 1, kernel_size=2, stride=2, dimension=2)

i = s

s = conv2(s)

g = upconv2(s)

c = DP.concatenate_sparse_tensors(g, i, 1)

