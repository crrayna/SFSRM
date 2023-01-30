import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class WGTDataset(data.Dataset):

    def __init__(self, opt):
        super(WGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type'] #lmdb or img
        self.paths_WF, self.paths_GT = None, None
        self.sizes_WF, self.sizes_GT = None, None
        self.WF_env, self.GT_env = None, None  # environment for lmdb
        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_WF, self.sizes_WF = util.get_image_paths(self.data_type, opt['dataroot_WF'])

        assert self.paths_GT, 'Error: GT path is empty.'
        assert self.paths_WF, 'Error: WF path is empty.'

        if self.paths_WF and self.paths_GT:
            assert len(self.paths_WF) == len(
                self.paths_GT
            ), 'GT and WF datasets have different number of images - {}, {}.'.format(
                len(self.paths_WF), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.WF_env = lmdb.open(self.opt['dataroot_WF'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.WF_env is None):
                self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)

        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)

        # get WF image
        WF_path = self.paths_WF[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_WF[index].split('_')]
        else:
            resolution = None
        img_WF = util.read_img(self.WF_env, WF_path, resolution)
        '''read image by cv2 or from lmdb
        return: Numpy float32, HWC, BGR, [0,1]'''
        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)

                # using matlab imresize
                img_WF = util.imresize_np(img_GT, 1 / scale, True)

            H, W, C = img_WF.shape
            WF_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - WF_size))
            rnd_w = random.randint(0, max(0, W - WF_size))
            img_WF = img_WF[rnd_h:rnd_h + WF_size, rnd_w:rnd_w + WF_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            # augmentation - flip, rotate
            img_GT, img_WF = util.augment([img_GT, img_WF], self.opt['use_flip'],
                                          self.opt['use_rot'])

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_WF = torch.from_numpy(np.ascontiguousarray(np.transpose(img_WF, (2, 0, 1)))).float()

        return {'GT': img_GT, 'LW': img_WF, 'GT_path': GT_path, 'WF_path': WF_path}

    def __len__(self):
        return len(self.paths_GT)
