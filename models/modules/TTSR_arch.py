import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()

        vgg_firstlayer = models.vgg19(pretrained=True).features[0]

        w1 = vgg_firstlayer.state_dict()['weight'][:, 0, :, :]
        w2 = vgg_firstlayer.state_dict()['weight'][:, 1, :, :]
        w3 = vgg_firstlayer.state_dict()['weight'][:, 2, :, :]
        w = w1 + w2 + w3
        w = w.unsqueeze(1)
        first_conv = nn.Conv2d(1, 64, 3, padding=(1, 1))
        first_conv.weight = torch.nn.Parameter(w, requires_grad=True)
        first_conv.bias = torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True)

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()

        self.slice2 = torch.nn.Sequential()

        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            if x == 0:
                self.slice1.add_module(str(x), first_conv)
            else:
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

    def forward(self, x):

        # rgb_range = 1
        # vgg_mean = 0.449
        # vgg_std = 0.226
        # x = (x - vgg_mean) / vgg_std
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SFE(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(1, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv21 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats * 2, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 2, n_feats)

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12), dim=1)))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        self.conv12 = conv1x1(n_feats, n_feats)
        self.conv13 = conv1x1(n_feats, n_feats)

        self.conv21 = conv3x3(n_feats, n_feats, 2)
        self.conv23 = conv1x1(n_feats, n_feats)

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge2 = conv3x3(n_feats * 3, n_feats)
        self.conv_merge3 = conv3x3(n_feats * 3, n_feats)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1(torch.cat((x1, x21, x31), dim=1)))
        x2 = F.relu(self.conv_merge2(torch.cat((x2, x12, x32), dim=1)))
        x3 = F.relu(self.conv_merge3(torch.cat((x3, x13, x23), dim=1)))

        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, n_feats)
        self.conv23 = conv1x1(n_feats, n_feats)
        self.conv_merge = conv3x3(n_feats * 3, n_feats)
        self.conv_tail1 = conv3x3(n_feats, n_feats // 2)
        self.conv_tail2 = conv1x1(n_feats // 2, 1)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))
        x = F.relu(self.conv_merge(torch.cat((x3, x13, x23), dim=1)))
        x = self.conv_tail1(x)
        x = self.conv_tail2(x)
        x = torch.clamp(x, -1, 1)

        return x


class MainNet(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale):
        super(MainNet, self).__init__()
        self.num_res_blocks = num_res_blocks  ### a list containing number of resblocks of different stages
        self.n_feats = n_feats

        self.SFE = SFE(self.num_res_blocks[0], n_feats, res_scale)

        ### stage11
        self.conv11_head = conv3x3(256 + n_feats, n_feats)
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
        self.conv11_tail = conv3x3(n_feats, n_feats)

        ### subpixel 1 -> 2
        self.conv12 = conv3x3(n_feats, n_feats * 4)
        self.ps12 = nn.PixelShuffle(2)

        ### stage21, 22
        # self.conv21_head = conv3x3(n_feats, n_feats)
        self.conv22_head = conv3x3(128 + n_feats, n_feats)

        self.ex12 = CSFI2(n_feats)

        self.RB21 = nn.ModuleList()
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB21.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB22.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))

        self.conv21_tail = conv3x3(n_feats, n_feats)
        self.conv22_tail = conv3x3(n_feats, n_feats)

        ### subpixel 2 -> 3
        self.conv23 = conv3x3(n_feats, n_feats * 4)
        self.ps23 = nn.PixelShuffle(2)

        ### stage31, 32, 33
        # self.conv31_head = conv3x3(n_feats, n_feats)
        # self.conv32_head = conv3x3(n_feats, n_feats)
        self.conv33_head = conv3x3(64 + n_feats, n_feats)

        self.ex123 = CSFI3(n_feats)

        self.RB31 = nn.ModuleList()
        self.RB32 = nn.ModuleList()
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB31.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB32.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))
            self.RB33.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                      res_scale=res_scale))

        self.conv31_tail = conv3x3(n_feats, n_feats)
        self.conv32_tail = conv3x3(n_feats, n_feats)
        self.conv33_tail = conv3x3(n_feats, n_feats)

        self.merge_tail = MergeTail(n_feats)

    def forward(self, x, S=None, T_lv3=None, T_lv2=None, T_lv1=None):
        ### shallow feature extraction
        x = self.SFE(x)

        ### stage11
        x11 = x

        ### soft-attention
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res)  # F.relu(self.conv11_head(x11_res))
        x11_res = x11_res * S
        x11 = x11 + x11_res

        x11_res = x11

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        ### stage21, 22
        x21 = x11
        x21_res = x21
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))

        ### soft-attention
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res)  # F.relu(self.conv22_head(x22_res))
        x22_res = x22_res * F.interpolate(S, scale_factor=2, mode='bicubic')
        x22 = x22 + x22_res

        x22_res = x22

        x21_res, x22_res = self.ex12(x21_res, x22_res)

        for i in range(self.num_res_blocks[2]):
            x21_res = self.RB21[i](x21_res)
            x22_res = self.RB22[i](x22_res)

        x21_res = self.conv21_tail(x21_res)
        x22_res = self.conv22_tail(x22_res)
        x21 = x21 + x21_res
        x22 = x22 + x22_res

        ### stage31, 32, 33
        x31 = x21
        x31_res = x31
        x32 = x22
        x32_res = x32
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))

        ### soft-attention
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res)  # F.relu(self.conv33_head(x33_res))
        x33_res = x33_res * F.interpolate(S, scale_factor=4, mode='bicubic')
        x33 = x33 + x33_res

        x33_res = x33

        x31_res, x32_res, x33_res = self.ex123(x31_res, x32_res, x33_res)

        for i in range(self.num_res_blocks[3]):
            x31_res = self.RB31[i](x31_res)
            x32_res = self.RB32[i](x32_res)
            x33_res = self.RB33[i](x33_res)

        x31_res = self.conv31_tail(x31_res)
        x32_res = self.conv32_tail(x32_res)
        x33_res = self.conv33_tail(x33_res)
        x31 = x31 + x31_res
        x32 = x32 + x32_res
        x33 = x33 + x33_res
        x = self.merge_tail(x31, x32, x33)

        return x

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        return S, T_lv3, T_lv2, T_lv1

class TTSR(nn.Module):
    def __init__(self, num_res_blocks='16+16+8+4',n_feats=64,res_scale=1):
        super(TTSR, self).__init__()
        self.num_res_blocks = list( map(int, num_res_blocks.split('+')) )
        self.MainNet = MainNet(num_res_blocks=self.num_res_blocks, n_feats=n_feats,
            res_scale=res_scale)
        self.LTE      = LTE(requires_grad=True)
        self.LTE_copy = LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())

            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)



        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)


        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)
        return sr, S, T_lv3, T_lv2, T_lv1
