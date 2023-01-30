import torch
import torch.nn as nn
import torch.nn.functional as F

def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad,\
            "nn criterions don't compute the gradient w.r.t targets"

def FFT_trans(x, eps):
    vF = torch.fft.fft2(x, norm='ortho')
    vF = torch.fft.fftshift(vF)
    vF = torch.stack([vF.real, vF.imag], -1)
    # get real part
    #vR = vF[:,:,33:33+192-1,33:33+192-1, 0]
    # get the imaginary part
    vF[:,:,96:96+64-1,96:96+64-1, :] = 0
    #vI = vF - vF[:,:,33:33+192-1,33:33+192-1, 1]
    #vR = vF - vF[:,:,33:33+192-1,33:33+192-1, 0]
    #vI = vF[:,:,65:193,65:193, 1]
    vR = vF[:,:,:,:, 0]
    vI = vF[:,:,:,:, 1]
    out_amp = torch.add(torch.pow(vR, 2), torch.pow(vI, 2))
    out_amp = torch.sqrt(out_amp + eps)
    out_pha = torch.atan2(vR,(vI + eps))
    return out_amp, out_pha


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.eps = 1e-7


    def forward(self, SR, GT):
        _assert_no_grad(GT)
        real_fft_amp,  real_fft_pha = FFT_trans(GT,self.eps)
        fake_fft_amp,  fake_fft_pha = FFT_trans(SR,self.eps)
        amp_dis = real_fft_amp - fake_fft_amp
        pha_dis = real_fft_pha - fake_fft_pha
        fft_dis = (torch.pow(amp_dis,2) + torch.pow(pha_dis,2) + self.eps).sqrt()  #  +
        fftloss = fft_dis.mean()
        return fftloss
