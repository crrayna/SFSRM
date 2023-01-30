import torch
import logging
import models.modules.RCAN_arch as RCAN_arch
import models.modules.RNAN_arch as RNAN_arch
import models.modules.UAN_arch as UAN_arch
import models.modules.TTSR_arch as TTSR_arch
import models.modules.DEFIAN_arch as DEFIAN_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.RFB_RRDB_arch as RFB_RRDB_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'RCAN':
        netG = RCAN_arch.RCAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                      scale=opt_net['scale'])
    elif which_model == 'TTSR':
        netG = TTSR_arch.TTSR()
    elif which_model == 'DEFIAN':
        netG = DEFIAN_arch.DEFIAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                      scale=[opt_net['scale']])
    elif which_model == 'UAN':
        netG = UAN_arch.UAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                      scale=opt_net['scale'])
    elif which_model == 'RNAN':
        netG = RNAN_arch.RNAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                      scale=opt_net['scale'])
    elif which_model == 'RRDBNet':

        netG = RRDBNet_arch.RRDBNet(in_nc= opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'RFB_RRDB':

        netG = RFB_RRDB_arch.RFB_RRDB(in_nc= opt_net['in_nc'], out_nc=opt_net['out_nc'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_256':
        netD = SRGAN_arch.Discriminator_VGG_256(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'Unet':
        netD = SRGAN_arch.UnetD()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

def define_D2(opt):
    opt_net = opt['network_D2']
    which_model = opt_net['which_model_D2']

    if which_model == 'patch_GAN':
        netD2 = SRGAN_arch.patch_GAN(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD2

#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF


