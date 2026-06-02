import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.archs.segment.hrseg_model import create_hrnet
from models.loss import CharbonnierLoss, VGGLoss, SSIM, ContrastLoss, CLIPLOSS

logger = logging.getLogger('base')


class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1

        train_opt = opt['train']

        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)

        if opt['seg']:
            self.seg_model = create_hrnet().cuda()
            self.seg_model.eval()
        else:
            self.seg_model = None

        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            loss_type = train_opt['pixel_criterion']

            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError()

            self.is_vgg_loss = train_opt['vgg_loss']
            self.l_pix_w = train_opt['pixel_weight']

            self.c_weight = train_opt['c_weight']
            self.ssim_weight = train_opt['ssim_weight']
            self.vgg_weight = train_opt['vgg_weight']
            self.clip_step = train_opt['clip_step']
            self.clip_weight = train_opt['clip_weight']
            self.grad_clip_norm = train_opt['grad_clip_norm']

            self.cri_pix_ill = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum').to(self.device)
            self.con_loss = ContrastLoss().to(self.device)
            self.cri_vgg = VGGLoss().to(self.device)
            self.ssim_loss = SSIM().to(self.device)
            self.l1_loss = torch.nn.L1Loss().to(self.device)
            self.clip_loss = CLIPLOSS().to(self.device)

            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                optim_params = [
                    {
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

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
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def combine_elements(self, arr1, arr2):
        combine_dict = {}
        for i in range(len(arr1)):
            combine_dict[i] = {arr1[i], arr2[i]}
        return list(combine_dict.values())

    def feed_data(self, data, need_GT):
        self.var_L = data['LQs'].to(self.device)
        self.neg_H = data['NEG'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        if self.seg_model is not None:
            self.seg_map, self.seg_feature = self.seg_model(self.real_H)
        else:
            self.seg_map, self.seg_feature = None, None

    def set_params_lr_zero(self):
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        c_loss = self.con_loss(self.var_L, self.real_H, self.neg_H, self.fake_H) * self.c_weight
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_ssim = (1 - self.ssim_loss(self.fake_H, self.real_H)) * self.ssim_weight

        vgg_loss_state = False
        if self.is_vgg_loss:
            l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * self.vgg_weight
            vgg_loss_state = True

        clip_loss_state = False
        if self.seg_map is not None and step % self.clip_step == 0:
            l_clip = self.clip_loss(self.seg_map, self.fake_H, self.real_H) * self.clip_weight
            if vgg_loss_state:
                l_final = l_pix + l_ssim + l_clip + c_loss + l_vgg
            else:
                l_final = l_pix + l_ssim + l_clip + c_loss
            clip_loss_state = True
        else: