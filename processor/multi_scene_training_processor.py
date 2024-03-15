# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp


def do_multi_scene_train(cfg,
                         model,
                         center_criterion,
                         train_loader,
                         val_loader_dict,
                         optimizer,
                         optimizer_center,
                         scheduler,
                         loss_fn,
                         len_query_list,
                         local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("VersReID.train")

    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            raise NotImplementedError

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator_dict = {}
    for i, name in enumerate(cfg.DATASETS.COMBINE_NAMES):
        evaluator_dict[name] = R1_mAP_eval(len_query_list[i], max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, dataset=name)

    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for name, evaluator in evaluator_dict.items():
            evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, pid, target_cam, target_view, dataset_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            dataset_ids = dataset_ids.to(device)
            with amp.autocast(enabled=True):
                gt_feat, fc_feat = model(img, target, cam_label=target_cam, view_label=target_view,
                                         dataset_label=dataset_ids, forward_aux=False)

                reid_loss = loss_fn(fc_feat, gt_feat, target, target_cam)

            scaler.scale(reid_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if cfg.MODEL.EMA and (n_iter + 1) % cfg.MODEL.EMA_S == 0:
                model.ema_update()

            if isinstance(fc_feat, list):
                fc_feat = fc_feat[0]
            acc = (fc_feat.max(1)[1] == target).float().mean()

            loss_meter.update(reid_loss.item(), img.shape[0])
            acc_meter.update(acc, 1.0)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Reid Loss: {:.3f},"
                    " Reid Acc: {:.3f},"
                    " Base Lr: {:.2e}".format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                                              acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_epoch = end_time - start_time
        logger.info("Epoch {} done. Time per epoch: {:.3f}[s]".format(epoch, time_per_epoch))

        if epoch % checkpoint_period == 0:
            if not cfg.MODEL.EMA:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                torch.save(model.ema_model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ema_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if not cfg.MODEL.EMA:
                model.eval()
                logger.info("------------------------------------")
                for dataset_id, (name, val_loader) in enumerate(val_loader_dict.items()):
                    logger.info('Evaluating Dataset ' + name)
                    dataset_type = int(cfg.DATASETS.COMBINE_TYPE[dataset_id])
                    for n_iter, (img, pid, camid, camids, vid, target_view, imgpath) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view, dataset_label=dataset_type)
                            evaluator_dict[name].update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator_dict[name].compute()
                    logger.info("Validation Results for " + name + " - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    logger.info("------------------------------------")
            else:
                model.model.eval()
                model.ema_model.eval()
                # test for model
                logger.info("------------------------------------")
                for name, val_loader in val_loader_dict.items():
                    logger.info('Evaluating Dataset ' + name)
                    for n_iter, (img, pid, camid, camids, vid, target_view, imgpath) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model.forward_model(img, cam_label=camids, view_label=target_view)
                            evaluator_dict[name].update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator_dict[name].compute()
                    logger.info("Validation Results for " + name + " - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    logger.info("------------------------------------")

                # test for ema model
                for name, evaluator in evaluator_dict.items():
                    evaluator.reset()
                logger.info("----------------EMA----------------")
                for name, val_loader in val_loader_dict.items():
                    logger.info('Evaluating Dataset ' + name)
                    for n_iter, (img, pid, camid, camids, vid, target_view, imgpath) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model.forward_ema_model(img, cam_label=camids, view_label=target_view)
                            evaluator_dict[name].update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator_dict[name].compute()
                    logger.info("Validation Results for EMA model " + name + " - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    logger.info("------------------------------------")
