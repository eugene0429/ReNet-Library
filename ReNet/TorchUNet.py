import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from EnvioX.data_processing import DataProcessing as DP # type: ignore

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