import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Copied from https://github.com/clovaai/stargan-v2/blob/master/core/model.py#L23
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2), down_sample=False, normalize=False):
        super().__init__()
        self.activation = activation
        self.normalize = normalize
        self.down_sample = down_sample
        self.change_dim = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.change_dim:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.change_dim:
            x = self.conv1x1(x)
        if self.down_sample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        if self.down_sample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


# original ADAIN : linear feature and linear style
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # orig : 64 ---> (FC) ---> num_features * 2
        # we need to 64x8x8 ---> num_features 
        # mapgan : conv(3, 1, 1) => result same dimension as feature
        self.fc = nn.Linear(style_dim, num_features * 2)


    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        
        # gamma, betta : num_features, num_features
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

# ADAIN2D : linear feature and linear style
class AdaIN2D(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # x : [0] 512x8x8 --> 512x16x16
        # s : [0] 64x8x8

        # orig : 64 ---> (FC) ---> num_features * 2
        # we need to 64x8x8 ---> num_features 
        # mapgan : conv(3, 1, 1) => result same dimension as feature
        #self.fc = nn.Linear(style_dim, num_features * 2)

        self.beta = nn.Conv2d(style_dim, num_features, 3, 1, 1)
        self.gamma = nn.Conv2d(style_dim, num_features, 3, 1, 1)
        self.num_features = num_features


    def forward(self, x, s):
        #h = self.fc(s)
        # s -> syle_dim -> num_features * 2 -> [B, num_featurs, 1, 1]
        #h = h.view(h.size(0), h.size(1), 1, 1)
        gamma = self.gamma(s)
        beta = self.beta(s)
        
        # gamma, betta : num_features, num_features
        #gamma, beta = torch.chunk(h, chunks=2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0, activation=nn.LeakyReLU(0.2), up_sample=False, use_type='2D'):
        super().__init__()
        self.w_hpf = w_hpf
        self.activation = activation
        self.up_sample = up_sample
        self.change_dim = dim_in != dim_out
        self.use_type = use_type
        self._build_weights(dim_in, dim_out, style_dim)
        

        # ------------------------
        self.gamma = []
        self.beta = []

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        
        if self.use_type == '2D':
            self.norm1 = AdaIN2D(style_dim, dim_in)
            self.norm2 = AdaIN2D(style_dim, dim_out)
        else:
            self.norm1 = AdaIN(style_dim, dim_in)
            self.norm2 = AdaIN(style_dim, dim_out)

        if self.change_dim:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.change_dim:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        if self.use_type == '2D':
            s = F.interpolate(s, size=x.size(2), mode='nearest')    # ---
        x = self.norm1(x, s)
        x = self.activation(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if self.use_type == '2D':
                s = F.interpolate(s, size=x.size(2), mode='nearest') # ----
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):

        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


