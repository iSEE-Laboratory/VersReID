# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
from utils.logger import setup_logger
from datasets import make_joint_test_dataloader
from model import make_model
import torch
import os
import argparse
from config import cfg
from utils.metrics import R1_mAP_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Multi-scene Testing")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("--name", default=None, help="path to config file", type=str)
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

    logger = setup_logger("VersReID", output_dir, if_train=False, file_name=args.name)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    val_loader, num_classes_list, total_classes, total_query, total_gallery = make_joint_test_dataloader(cfg)

    device = "cuda"
    cfg.defrost()
    model = make_model(cfg, num_class=total_classes, camera_num=0, view_num=0)
    ckpt = torch.load(cfg.TEST.WEIGHT, 'cpu')
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'classifier' not in k:
            new_ckpt[k] = v
    msg = model.load_state_dict(new_ckpt, strict=False)
    print(msg)
    model.to(device)

    evaluator = R1_mAP_eval(total_query,
                            max_rank=50,
                            feat_norm=cfg.TEST.FEAT_NORM,
                            dataset=cfg.DATASETS.NAMES,
                            reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    model.eval()
    logger.info("------------------------------------")
    for n_iter, (img, pid, camids, camids_b, viewids, viewids_b, dataset_ids_b) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.AUX_LOSS:
                dataset_ids_b = -1
            feat = model(img, cam_label=None, view_label=None, dataset_label=dataset_ids_b)
            evaluator.update((feat, pid, camids))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results for ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    torch.cuda.empty_cache()
    logger.info("------------------------------------")
