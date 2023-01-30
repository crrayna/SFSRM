import torch.nn as nn
import torch, math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pylab
from PIL import Image




def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output


class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # # Freeze the MeanShift layer
        # for params in self.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian


class rcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(rcab_block, self).__init__()

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        residue = self.block(x)
        chnlatt = F.adaptive_avg_pool2d(residue, 1)
        chnlatt = self.calayer(chnlatt)
        output = x + residue * chnlatt

        return output


class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output


class SingleModule(nn.Module):
    def __init__(self, n_channels, n_blocks, act, attention):
        super(SingleModule, self).__init__()
        res_blocks = [rcab_block(n_channels=n_channels, kernel=3, activation=act) for _ in range(n_blocks)]
        self.body_block = nn.Sequential(*res_blocks)
        self.attention = attention
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def forward(self, x):
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = resin

        output = resout + x

        return output


class DEFIAN(nn.Module):
    def __init__(self, in_nc, out_nc, n_channels=64, n_blocks=10, n_modules=5, act=nn.ReLU(True), attention=True, scale=(2, 3, 4)):
        super(DEFIAN, self).__init__()
        self.n_modules = n_modules
        self.input = nn.Conv2d(in_channels=in_nc, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True)
        if n_modules == 1:
            self.body = nn.Sequential(SingleModule(n_channels, n_blocks, act, attention))
        else:
            self.body = nn.Sequential(*[SingleModule(n_channels, n_blocks, act, attention) for _ in range(n_modules)])

        self.tail = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

        self.upscale = nn.ModuleList([UpScale(n_channels=n_channels, scale=s, act=False) for s in scale])

        self.output = nn.Conv2d(in_channels=n_channels, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        body_input = self.input(x)
        body_output = self.body(body_input)
        if self.n_modules == 1:
            sr_high = self.upscale[0](body_output)
        else:
            sr_high = self.upscale[0](self.tail(body_output) + body_input)
        results = self.output(sr_high)

        return results



class UpScale(nn.Sequential):
    def __init__(self, n_channels, scale, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4 * n_channels, kernel_size=3, stride=1,
                                        padding=1, bias=bias))
                layers.append(nn.PixelShuffle(2))
                if bn: layers.append(nn.BatchNorm2d(n_channels))
                if act: layers.append(act)
        elif scale == 3:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=9 * n_channels, kernel_size=3, stride=1,
                                    padding=1, bias=bias))
            layers.append(nn.PixelShuffle(3))
            if bn: layers.append(nn.BatchNorm2d(n_channels))
            if act: layers.append(act)
        else:
            raise NotImplementedError

        super(UpScale, self).__init__(*layers)
