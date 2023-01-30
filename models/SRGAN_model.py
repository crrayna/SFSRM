import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
'''Base class for all neural network modules'''
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss
from models.modules.MS_SSIM_L1 import MSSSIML1_Loss, SSIM
from models.modules.FFT_loss import FFTLoss
# from models.modules.lpips_pytorch.modules.lpips import LPIPS
# from models.modules.DISTS_pytorch import DISTS


logger = logging.getLogger('base')

class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            if train_opt['gan_weight'] > 0:
                self.netD.train()
    

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'ms_ssim_l1':
                    self.cri_pix = MSSSIML1_Loss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            # G FFT loss
            if train_opt['fft_weight'] > 0:
                self.cri_fft = FFTLoss().to(self.device)
                self.l_fft_w = train_opt['fft_weight']
            else:
                logger.info('Remove fft loss.')
                self.cri_fft = None

            # G Fm loss
            if train_opt['fm_weight'] > 0:
                l_fm_type = train_opt['fm_criterion']
                if l_fm_type == 'l1':
                    self.cri_fm = nn.L1Loss().to(self.device)
                elif l_fm_type == 'l2':
                    self.cri_fm = nn.MSELoss().to(self.device)
                elif l_fm_type == 'ms_ssim_l1':
                    self.cri_fm = MSSSIML1_Loss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fm_type))
                self.l_fm_w = train_opt['fm_weight']
            else:
                logger.info('Remove fm loss.')
                self.cri_fm = None

                # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'ssim_l1':
                    self.cri_fea = MSSSIML1_Loss().to(self.device)    
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None

            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.cri_l1 = nn.L1Loss().to(self.device)
            self.cri_l2 = nn.MSELoss().to(self.device)
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed




    def feed_data(self, data, need_GT=True):
        self.var_L = data['LW'].to(self.device)  # LW
        if need_GT:
            self.var_H = data['GT'].to(self.device)
            self.var_H_3 = self.var_H.repeat(1, 3, 1, 1)
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        self.fake_H_3 = self.fake_H.repeat(1, 3, 1, 1)
        
        
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fft:  # fft loss
                l_g_fft = self.l_fft_w * self.cri_fft(self.fake_H, self.var_H)
                l_g_total += l_g_fft

            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H_3).detach()
                fake_fea = self.netF(self.fake_H_3)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
           
            if self.l_gan_w > 0:
                e_S, d_S, e_Ss, d_Ss = self.netD(self.fake_H)
                _, _, e_Hs, d_Hs = self.netD(self.var_ref)
                l_g_gan = self.l_gan_w * 0.5 * (self.cri_gan(e_S, True) + self.cri_gan(d_S, True))
                l_g_total += l_g_gan
        
                if self.cri_fm:
                    l_g_fms = 0
                    for f in range(6):
                        l_g_fms += self.cri_fm(e_Ss[f], e_Hs[f])
                        l_g_fms += self.cri_fm(d_Ss[f], d_Hs[f])
                    l_g_fm = l_g_fms / 6 * self.l_fm_w
                    l_g_total += l_g_fm

            l_g_total.backward()
            self.optimizer_G.step()


        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total_D = 0
        e_S, d_S, _, _ = self.netD(self.fake_H.detach())
        e_H, d_H, _, _ = self.netD(self.var_ref)
        l_d_real_e = self.cri_gan(e_H, True)
        l_d_fake_e = self.cri_gan(e_S, False)

        l_d_real_d = self.cri_gan(d_H, True)
        l_d_fake_d = self.cri_gan(d_S, False)
        l_d_real_D = l_d_real_e + l_d_real_d
        l_d_total_D += l_d_real_D

        fake_H_CutMix = self.fake_H.detach().clone()

        #probability of doing cutmix
        p_mix = step /100000 #100000
        if p_mix > 0.5:
            p_mix = 0.5

        if torch.rand(1) <= p_mix:
            #n_mix += 1
            r_mix = torch.rand(1)  # real/fake ratio

            #def rand_bbox(self, size, lam):
            B = fake_H_CutMix.size()[0]
            C = fake_H_CutMix.size()[1]
            W = fake_H_CutMix.size()[2]
            H = fake_H_CutMix.size()[3]

            cut_rat = np.sqrt(1. - r_mix)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

        #         # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            fake_H_CutMix[:, :, bbx1:bbx2, bby1:bby2] = self.var_H[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (fake_H_CutMix.size()[-1] * fake_H_CutMix.size()[-2]))

            e_mix, d_mix, _, _ = self.netD(fake_H_CutMix)
            mask_true = torch.zeros(B,C,W,H)
            mask_true[:, :, bbx1:bbx2, bby1:bby2]=1
            mask_false= 1 - mask_true
            self.mask_true = mask_true.to(self.device)
            self.mask_false = mask_false.to(self.device)
            l_d_fake_e = self.cri_gan(e_mix, False)
            #l_d_fake_d = self.cri_gan(((1-2*self.mask)*d_mix), False)
            l_d_fake_d_r = self.cri_gan( d_mix * (self.mask_true),True)
            l_d_fake_d_f = self.cri_gan( d_mix * (self.mask_false), False)
            l_d_fake_d = l_d_fake_d_r * (1 - r_mix) + l_d_fake_d_f * (r_mix)

            d_S[:, :, bbx1:bbx2, bby1:bby2] = d_H[:, :, bbx1:bbx2, bby1:bby2]

            loss_d_cons = self.cri_l1(d_mix, d_S)

            l_d_total_D += loss_d_cons
        #
        l_d_fake_D = l_d_fake_e + l_d_fake_d
        l_d_total_D += l_d_fake_D
        l_d_total_D.backward()
        self.optimizer_D.step()
        
        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fm:
                self.log_dict['l_g_fm'] = l_g_fm.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            if self.cri_fft:
                self.log_dict['l_g_fft'] = l_g_fft.item()
            if self.l_gan_w >0:
                self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real_D.item()
        self.log_dict['l_d_fake'] = l_d_fake_D.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)
            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])


    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.l_gan_w >0:
            self.save_network(self.netD, 'D', iter_step)

