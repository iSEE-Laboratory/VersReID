# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
import torch
import torch.nn as nn
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BuildTransformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(BuildTransformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if 'small' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        else:
            self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num,
                                                        view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.PRETRAIN_CHOICE != 'none':
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.base.fc = nn.Identity()

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None, dataset_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        base_dict = {}
        for k, v in param_dict.items():
            if 'classifier' in k or 'fc.' in k:
                continue
            if 'base' in k:
                base_dict[k.replace('module.', '').replace('base.', '')] = v
                continue
            else:
                self.state_dict()[k.replace('module.', '')].copy_(v)
        self.base.load_param(model_path=None, param_dict=base_dict)
        print('Loading pretrained model from {}'.format(trained_path))


class BuildMultiSceneTransformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(BuildMultiSceneTransformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_scenes = len(set(cfg.DATASETS.COMBINE_TYPE))
        if 'small' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        else:
            self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num,
                                                        view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        scene_nums=self.num_scenes,
                                                        scene_prompt_type=cfg.MODEL.SCENE_PROMPT_TYPE,
                                                        scene_prompt_nums=cfg.MODEL.SCENE_PROMPT_NUMS,
                                                        aux_loss=cfg.MODEL.AUX_LOSS,
                                                        aux_prompt_nums=cfg.MODEL.AUX_PROMPT_NUMS)

        if cfg.MODEL.PRETRAIN_CHOICE != 'none':
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

        self.base.fc = nn.Identity()

        assert isinstance(num_classes, int)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.aux_loss = cfg.MODEL.AUX_LOSS

    def forward(self, x, label=None, cam_label=None, view_label=None, dataset_label=None,
                forward_aux=False):  # dataset_label is the type of scene
        if self.training:
            gt_feat = self.base(x, cam_label=cam_label, view_label=view_label, dataset_label=dataset_label)
            bn_feat = self.bottleneck(gt_feat)
            fc_feat = self.classifier(bn_feat)
            if self.aux_loss and forward_aux:
                feat = self.base(x, cam_label=cam_label, view_label=view_label, dataset_label=-1)
                return gt_feat, fc_feat, feat
            else:
                return gt_feat, fc_feat
        # for test
        else:
            feat = self.base(x, cam_label=cam_label, view_label=view_label, dataset_label=dataset_label)
            bn_feat = self.bottleneck(feat)
            if self.neck_feat == 'after':
                return bn_feat
            else:
                return feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        base_dict = {}
        for k, v in param_dict.items():
            if 'classifier' in k or 'fc.' in k:
                continue
            if 'base' in k:
                base_dict[k.replace('module.', '').replace('base.', '')] = v
                continue
            else:
                self.state_dict()[k.replace('module.', '')].copy_(v)
        self.base.load_param(model_path=None, param_dict=base_dict)
        print('Loading pretrained model from {}'.format(trained_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
}


class EMA(nn.Module):
    def __init__(self, model, momentum=0.999):
        super(EMA, self).__init__()
        self.model = model
        # make a copy of the model for accumulating moving average of weights
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.momentum = momentum
        print('EMA model with m =', self.momentum)
        for param in self.ema_model.parameters():
            param.requires_grad = False  # not update by gradient

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        if self.training:
            return self.model(x, label, cam_label, view_label)

    def ema_update(self):
        self.ema_model.eval()
        # print('EMA updating...')
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward_model(self, x, label=None, cam_label=None, view_label=None):
        assert not self.model.training
        return self.model(x, label, cam_label, view_label)

    def forward_ema_model(self, x, label=None, cam_label=None, view_label=None):
        assert not self.ema_model.training
        return self.ema_model(x, label, cam_label, view_label)


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.TYPE == 'TR':
        if cfg.MODEL.MULTI_SCENES:
            model = BuildMultiSceneTransformer(num_class,
                                               camera_num,
                                               view_num,
                                               cfg,
                                               __factory_T_type)
            print('===========Build Multi-scene Transformer===========')
        else:
            model = BuildTransformer(num_class,
                                     camera_num,
                                     view_num,
                                     cfg,
                                     __factory_T_type)
            print('===========Build Transformer===========')
    else:
        raise NotImplementedError
    print(model)
    if cfg.MODEL.EMA:
        model = EMA(model,
                    cfg.MODEL.EMA_M)
        print('===========EMA Model===========')
    return model
