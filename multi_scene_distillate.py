# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
from utils.logger import setup_logger
from datasets import make_combine_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_multi_scene_distillate
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Multi-scene Distillation")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("VersReID", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        raise NotImplementedError

    if cfg.MODEL.EMA:
        raise NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader_dict, num_classes_list, total_classes, len_query_list = make_combine_dataloader(cfg)

    cfg.defrost()
    dpr = cfg.MODEL.DROP_PATH
    print('===Teacher===')
    cfg.MODEL.DROP_PATH = 0.0
    teacher = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)
    ckpt = torch.load(cfg.MODEL.PRETRAIN_PATH, 'cpu')
    msg = teacher.load_state_dict(ckpt)
    teacher.eval()
    print('Freeze Teacher...')
    for param in teacher.parameters():
        param.requires_grad = False
    print(msg)

    print('===Student===')
    cfg.MODEL.AUX_LOSS = True
    cfg.MODEL.DROP_PATH = dpr
    student = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)
    msg = student.load_state_dict(ckpt, False)
    print(msg)

    loss_func, center_criterion = make_loss(cfg, num_classes=total_classes)

    optimizer, optimizer_center = make_optimizer(cfg, student, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_multi_scene_distillate(cfg,
                              teacher,
                              student,
                              center_criterion,
                              train_loader,
                              val_loader_dict,
                              optimizer,
                              optimizer_center,
                              scheduler,
                              loss_func,
                              len_query_list,
                              args.local_rank)
