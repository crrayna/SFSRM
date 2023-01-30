import torch
import torch.nn as nn
import torch.nn.functional as F

class TPerceptualLoss(nn.Module):
    def __init__(self, use_S=True, type='l2'):
        super(TPerceptualLoss, self).__init__()
        self.use_S = use_S
        self.type = type

    def gram_matrix(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, h * w)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def forward(self, map_lv3, map_lv2, map_lv1, S, T_lv3, T_lv2, T_lv1):
        ### S.size(): [N, 1, h, w]
        if (self.use_S):
            S_lv3 = torch.sigmoid(S)
            S_lv2 = torch.sigmoid(F.interpolate(S, size=(S.size(-2) * 2, S.size(-1) * 2), mode='bicubic'))
            S_lv1 = torch.sigmoid(F.interpolate(S, size=(S.size(-2) * 4, S.size(-1) * 4), mode='bicubic'))
        else:
            S_lv3, S_lv2, S_lv1 = 1., 1., 1.

        if (self.type == 'l1'):
            loss_texture = F.l1_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.l1_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.l1_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.
        elif (self.type == 'l2'):
            loss_texture = F.mse_loss(map_lv3 * S_lv3, T_lv3 * S_lv3)
            loss_texture += F.mse_loss(map_lv2 * S_lv2, T_lv2 * S_lv2)
            loss_texture += F.mse_loss(map_lv1 * S_lv1, T_lv1 * S_lv1)
            loss_texture /= 3.

        return loss_texture
