# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
from utils.logger import setup_logger
from datasets import make_combine_dataloader
from model import make_model
import torch
import os
import argparse
from config import cfg
from utils.metrics import R1_mAP_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Single-scene Testing")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("VersReID", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader_dict, num_classes_list, total_classes, len_query_list = make_combine_dataloader(cfg)

    device = "cuda"
    model = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)
    ckpt = torch.load(cfg.TEST.WEIGHT, 'cpu')
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'classifier' not in k:
            new_ckpt[k] = v
    msg = model.load_state_dict(new_ckpt, strict=False)
    print(msg)
    model.to(device)

    evaluator_dict = {}
    for i, name in enumerate(cfg.DATASETS.COMBINE_NAMES):
        evaluator_dict[name] = R1_mAP_eval(len_query_list[i], max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, dataset=name)
        evaluator_dict[name].reset()

    sum_r_1 = 0.0
    sum_map = 0.0

    model.eval()
    logger.info("------------------------------------")
    for dataset_id, (name, val_loader) in enumerate(val_loader_dict.items()):
        logger.info('Evaluating Dataset ' + name)
        dataset_type = -1 if cfg.MODEL.AUX_LOSS else int(cfg.DATASETS.COMBINE_TYPE[dataset_id])
        if cfg.TEST.ASSIGN_SCENE != -1:
            dataset_type = cfg.TEST.ASSIGN_SCENE
        for n_iter, (img, pid, camid, camids, vid, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)
                feat = model(img, cam_label=camids, view_label=target_view, dataset_label=dataset_type)
                evaluator_dict[name].update((feat, pid, camid))
        cmc, mAP, _, _, _, _, _ = evaluator_dict[name].compute()
        logger.info("Validation Results for ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        sum_r_1 += cmc[0]
        sum_map += mAP
        torch.cuda.empty_cache()
        logger.info("------------------------------------")
    logger.info("Avg Rank-1: {:.1%} | Avg mAP: {:.1%}".format(sum_r_1 / len(val_loader_dict),
                                                              sum_map / len(val_loader_dict)))
