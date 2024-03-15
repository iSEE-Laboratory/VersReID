# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import CombineImageDataset
from .market1501 import Market1501
from .msmt17 import MSMT17
from .occluded_duke import OccDukeMTMCreID
from .prcc import PRCC
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
    imgs, pids, camids, viewids, datasetids, _ = zip(*batch)
    viewids_batch = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    datasetids = torch.tensor(datasetids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, viewids_batch, datasetids


def make_joint_test_dataloader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS

    assert cfg.DATASETS.NAMES == 'combine'
    dataset = __factory[cfg.DATASETS.NAMES](names=cfg.DATASETS.COMBINE_NAMES,
                                            roots=cfg.DATASETS.ROOTS,
                                            subset_type=cfg.DATASETS.COMBINE_TYPE,
                                            combine_pid=cfg.DATASETS.COMBINE_PID)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # classes for every scene
    num_classes_list = dataset.classes_train_subset
    # total classes
    total_classes = dataset.total_classes

    joint_query, joint_gallery = make_joint_test_set(dataset.query_dict, dataset.gallery_dict,
                                                     cfg.DATASETS.COMBINE_NAMES, cfg.DATASETS.COMBINE_TYPE)
    total_query = len(joint_query)
    total_gallery = len(joint_gallery)
    val_set = CombineImageDataset(joint_query + joint_gallery, val_transforms)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_collate_fn)
    return val_loader, num_classes_list, total_classes, total_query, total_gallery


def make_joint_test_set(query_dict, gallery_dict, name_list, dataset_type):
    pid_begin = 0
    camid_begin = 0
    joint_query = []
    joint_gallery = []
    for i, name in enumerate(name_list):
        query_set, gallery_set = query_dict[name], gallery_dict[name]
        new_query, new_gallery = [], []
        pid_relabel_dict = {}
        camid_relabel_dict = {}
        for item in query_set + gallery_set:
            pid, camid = item[1], item[2]
            if pid not in pid_relabel_dict:
                pid_relabel_dict[pid] = len(pid_relabel_dict) + pid_begin
            if camid not in camid_relabel_dict:
                camid_relabel_dict[camid] = len(camid_relabel_dict) + camid_begin
        for item in query_set:
            img_path, pid, camid, vid = item
            new_query.append((img_path, pid_relabel_dict[pid], camid_relabel_dict[camid], vid, dataset_type[i]))
        for item in gallery_set:
            img_path, pid, camid, vid = item
            new_gallery.append((img_path, pid_relabel_dict[pid], camid_relabel_dict[camid], vid, dataset_type[i]))

        pid_begin += len(pid_relabel_dict)
        camid_begin += len(camid_relabel_dict)

        joint_query += new_query
        joint_gallery += new_gallery

    return joint_query, joint_gallery
