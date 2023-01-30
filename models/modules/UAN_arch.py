import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import *
import math
from math import log2

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='mish',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mish':
            self.act = Mish()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
        
## Upsampling Attention Block (UAB)
class UAB(nn.Module):
    def __init__(self, features, phase, activation=None, M=3, r=16):
        super(UAB, self).__init__()
        self.convs = nn.ModuleList([])

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif self.activation == 'mish':
            self.act = Mish()

        for i in range(M):
            upsample = []
            for _ in range(phase):
                upsample.append(ConvBlock(features, 4 * features, kernel_size=3))
                upsample.append(nn.PixelShuffle(2))
                if self.activation: upsample.append(self.act)
            self.convs.append(nn.Sequential(*upsample))

        self.fc = nn.Linear(features, r)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(r, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


## Channel Attention (CA) Layer
class CALayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 1控制的是输出的size
        # feature channel downscale and upscale --> channel weight
        self.conv_du = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




## Spatial Attention (SA) Layer    
class SALayer(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return x * self.sigmoid(y)


## Residual Feature Attention Block (RFAB)
class RFAB(torch.nn.Module):
    def __init__(self, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=Mish(), res_scale=1):
        super(RFAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(torch.nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn: modules_body.append(torch.nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        modules_body.append(SALayer(3))
        self.body = torch.nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res


## Residual Group (RG)
class ResidualGroup(torch.nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RFAB(
                n_feat, kernel_size, bias=True, bn=False) \
            for _ in range(n_resblocks)]
        modules_body.append(ConvBlock(n_feat, n_feat, kernel_size))
        self.body = torch.nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

class UAN(nn.Module):
    def __init__(self, in_nc,out_nc, scale,num_features=64):
        super(UAN, self).__init__()
        self.phase = int(log2(scale))
    
        n_resgroups = 6
        n_resblocks = 10
        
        self.head = ConvBlock(in_nc, num_features, 3, 1, 1, activation=None, norm=None)
        
        necks = [
            ResidualGroup(num_features, kernel_size = 3, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        self.neck = nn.Sequential(*necks)
        
        self.body = UAB(features=num_features, phase=self.phase, activation=False, M=3)

        self.tail = ConvBlock(num_features, out_nc, 3, 1, 1, activation=None, norm=None)
        
            
    def forward(self, x):
        x = self.head(x)
        res = self.neck(x)
        x = x + res
        x = self.body(x)
        x = self.tail(x)
        
        return x
    