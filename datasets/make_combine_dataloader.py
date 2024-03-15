# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import CombineImageDataset, ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySamplerCombine, SceneBalanceSampler
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occluded_duke import OccDukeMTMCreID
from .prcc import PRCC
from .preprocessing import GlobalCJ, GlobalGS, GlobalBlur, LocalCJ, LocalGS, LocalBlur, ImageNetPolicy
from .combine import CombineDataset
from .celebrity import Celebrity
from .sysu_mm01 import SysuMM01
from .dslr_cuhk03 import DSLR_CUHK03

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'prcc': PRCC,
    'occluded_duke': OccDukeMTMCreID,
    'combine': CombineDataset,
    'celebrity': Celebrity,
    'sysumm01': SysuMM01,
    'dslr_cuhk03': DSLR_CUHK03
}


def train_collate_fn(batch):
    imgs, pids, camids, viewids, datasetids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    datasetids = torch.tensor(datasetids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, datasetids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids_batch = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, viewids_batch, img_paths


def make_combine_dataloader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS

    assert cfg.DATASETS.NAMES == 'combine'
    num_scenes = len(set(cfg.DATASETS.COMBINE_TYPE))
    dataset = __factory[cfg.DATASETS.NAMES](names=cfg.DATASETS.COMBINE_NAMES,
                                            roots=cfg.DATASETS.ROOTS,
                                            subset_type=cfg.DATASETS.COMBINE_TYPE,
                                            combine_pid=cfg.DATASETS.COMBINE_PID)

    total_iter = len(dataset.train) // cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_EPOCHS

    if not cfg.INPUT.AUTO_AUG:
        train_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                                      T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                                      T.Pad(cfg.INPUT.PADDING),
                                      T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                                      GlobalCJ(probability=cfg.INPUT.GCJ_PROB),  # Global Color Jitter
                                      GlobalGS(probability=cfg.INPUT.GGS_PROB),  # Global Gray scale
                                      GlobalBlur(probability=cfg.INPUT.GB_PROB),  # Global Blur
                                      LocalCJ(probability=cfg.INPUT.LCJ_PROB),  # Local Color Jitter
                                      LocalGS(probability=cfg.INPUT.LGS_PROB),  # Local Gray scale
                                      LocalBlur(probability=cfg.INPUT.LB_PROB),  # Local Blur
                                      T.ToTensor(),
                                      T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                                      RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1,
                                                    device='cpu')])
    else:
        train_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                                      T.Pad(cfg.INPUT.PADDING),
                                      T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                                      ImageNetPolicy(total_iter=total_iter),
                                      T.ToTensor(),
                                      T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                                      RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1,
                                                    device='cpu')])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    train_set = CombineImageDataset(dataset.train, train_transforms)

    # classes for every scene
    num_classes_list = dataset.classes_train_subset
    # total classes
    total_classes = dataset.total_classes
    # nums of images for every scene
    len_train_list = dataset.len_train_subset
    # nums of query for every testing set
    len_query_list = dataset.len_query_list

    if 'triplet' in cfg.DATALOADER.COMBINE_SAMPLER:
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  sampler=RandomIdentitySamplerCombine(dataset.train,
                                                                       cfg.SOLVER.IMS_PER_BATCH,
                                                                       cfg.DATALOADER.NUM_INSTANCE),
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn)
    elif cfg.DATALOADER.COMBINE_SAMPLER == 'scene_balance':
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  sampler=SceneBalanceSampler(dataset.train,
                                                              cfg.SOLVER.IMS_PER_BATCH,
                                                              cfg.DATALOADER.NUM_INSTANCE,
                                                              len_train_list,
                                                              num_scenes),
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn)
    elif cfg.DATALOADER.COMBINE_SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_dict = {}
    val_loader_dict = {}
    for name in cfg.DATASETS.COMBINE_NAMES:
        val_set_dict[name] = ImageDataset(dataset.query_dict[name] + dataset.gallery_dict[name],
                                          val_transforms)
        val_loader_dict[name] = DataLoader(val_set_dict[name],
                                           batch_size=cfg.TEST.IMS_PER_BATCH,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           collate_fn=val_collate_fn)

    return train_loader, val_loader_dict, num_classes_list, total_classes, len_query_list
