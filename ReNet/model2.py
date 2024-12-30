import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import EnvioX

class PruningLayer(ME.MinkowskiNetwork):
    def __init__(self, in_channels, D, alpha):
        super(PruningLayer, self).__init__(D)
        self.alpha = alpha
        self.likelihood_conv = ME.MinkowskiConvolution(in_channels, out_channels=1, kernel_size=1, stride=1, dimension=D)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x):
        likelihood_map = MF.sigmoid(self.likelihood_conv(x))
        mask = (likelihood_map.F >= self.alpha).squeeze()
        pruned_features = self.pruning(x, mask)
        return likelihood_map, pruned_features

class PruningLayer_(ME.MinkowskiNetwork):
    def __init__(self, D, stride):
        super(PruningLayer_, self).__init__(D)
        self.pruning = ME.MinkowskiPruning()
        self.stride = stride
    def forward(self, x):
        coords = x.C
        mask = (coords[:,-1] == 0).squeeze()
        x = self.pruning(x, mask)
        x = EnvioX.SP.reduce_dimension(x, self.stride)
        return x

class Conv(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(Conv, self).__init__(D)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = MF.relu(x)
        return x

class DownConv(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(DownConv, self).__init__(D)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=(2,2,2,1), dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = MF.relu(x)
        return x
    
class UpConv(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(UpConv, self).__init__(D)
        self.up_conv = ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.norm(x)
        x = MF.relu(x)
        return x

class BridgeConv(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(BridgeConv, self).__init__(D)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=(1,1,1,2), dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = MF.relu(x)
        return x

class FinalConv(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(FinalConv, self).__init__(D)
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, stride=1, dimension=D)
        self.norm = ME.MinkowskiBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class EncoderBox(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, stride):
        super(EncoderBox, self).__init__(D)
        self.conv = Conv(in_channels, out_channels, D)
        self.down_conv = DownConv(out_channels, out_channels, D)
        self.pruning = PruningLayer_(D, stride)

    def forward(self, x, check):
        x = self.conv(x)
        skip = x
        skip = self.pruning(skip)
        x = self.down_conv(x)
        if check:
            EnvioX.SP.print_sparse_tensor_num_coords_and_shape(x)
        return skip, x

class Bridge(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D):
        super(Bridge, self).__init__(D)
        self.conv = BridgeConv(in_channels, out_channels, D)
        self.up_conv = UpConv(out_channels, out_channels, D-1)

    def forward(self, x, check):
        x = self.conv(x)
        x = EnvioX.SP.reduce_dimension(x, 16)
        x = self.up_conv(x)
        if check:
            EnvioX.SP.print_sparse_tensor_num_coords_and_shape(x)
        return x

class DecoderBox(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, alpha, stride):
        super(DecoderBox, self).__init__(D)
        self.conv = Conv(in_channels, out_channels, D)
        self.pruning = PruningLayer(out_channels, D, alpha)
        self.up_conv = UpConv(out_channels, out_channels, D)
        self.stride = stride

    def forward(self, x, skip_connection, check):
        x = EnvioX.SP.concatenate_sparse_tensors(x, skip_connection, self.stride)
        x = self.conv(x)
        lh, x = self.pruning(x)
        x = self.up_conv(x)
        if check:
            EnvioX.SP.print_sparse_tensor_num_coords_and_shape(x)
        return lh, x

class FinalDecoderBox(ME.MinkowskiNetwork):
    def __init__(self, in_channels, mid_channels, out_channels, D, alpha, stride):
        super(FinalDecoderBox, self).__init__(D)
        self.conv1 = Conv(in_channels, mid_channels, D)
        self.pruning = PruningLayer(mid_channels, D, alpha)
        self.conv2 = FinalConv(mid_channels, out_channels, D)
        self.stride = stride

    def forward(self, x, skip_connection, check):
        x = EnvioX.SP.concatenate_sparse_tensors(x, skip_connection, self.stride)
        x = self.conv1(x)
        lh, x = self.pruning(x)
        x = self.conv2(x)
        if check:
            EnvioX.SP.print_sparse_tensor_num_coords_and_shape(x)
        return lh, x

class ReNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels, out_channels, D, alpha):
        super(ReNet, self).__init__(D)
        self.ch = [2, 4, 8, 16, 32]
        self.enc1 = EncoderBox(in_channels, in_channels * self.ch[0], D, 1) 
        self.enc2 = EncoderBox(in_channels * self.ch[0], in_channels * self.ch[1], D, 2)
        self.enc3 = EncoderBox(in_channels * self.ch[1], in_channels * self.ch[2], D, 4)
        self.enc4 = EncoderBox(in_channels * self.ch[2], in_channels * self.ch[3], D, 8)

        self.bridge = Bridge(in_channels * self.ch[3], in_channels * self.ch[4], D)

        self.dec4 = DecoderBox(in_channels * self.ch[4] + in_channels * self.ch[3], in_channels * self.ch[3], D-1, alpha, 8)
        self.dec3 = DecoderBox(in_channels * self.ch[3] + in_channels * self.ch[2], in_channels * self.ch[2], D-1, alpha, 4)
        self.dec2 = DecoderBox(in_channels * self.ch[2] + in_channels * self.ch[1], in_channels * self.ch[1], D-1, alpha, 2)
        self.dec1 = FinalDecoderBox(in_channels * self.ch[1] + in_channels * self.ch[0], in_channels * self.ch[0], out_channels, D-1, alpha, 1)

    def forward(self, x, check=False):
        skip1, x = self.enc1(x, check)
        skip2, x = self.enc2(x, check)
        skip3, x = self.enc3(x, check)
        skip4, x = self.enc4(x, check)

        x = self.bridge(x, check)

        lh1, x = self.dec4(x, skip4, check)
        lh2, x = self.dec3(x, skip3, check)
        lh3, x = self.dec2(x, skip2, check)
        lh4, x = self.dec1(x, skip1, check)

        return x, [lh1, lh2, lh3, lh4]
