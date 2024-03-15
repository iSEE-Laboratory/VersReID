# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.nn.functional as F


def do_multi_scene_distillate(cfg,
                              teacher,
                              student,
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

    if cfg.SOLVER.AUX_LOSS_TYPE == 'L2':
        aux_loss_fn = nn.MSELoss().cuda()
    elif cfg.SOLVER.AUX_LOSS_TYPE == 'L1':
        aux_loss_fn = nn.L1Loss().cuda()
    elif cfg.SOLVER.AUX_LOSS_TYPE == 'SmoothL1':
        aux_loss_fn = nn.SmoothL1Loss()
    elif cfg.SOLVER.AUX_LOSS_TYPE == 'KL':
        aux_loss_fn = nn.KLDivLoss()
    else:
        aux_loss_fn = None

    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        student.to(local_rank)
        teacher.to(local_rank)

        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            raise NotImplementedError

    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
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
        student.train()
        for n_iter, (img, pid, target_cam, target_view, dataset_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            dataset_ids = dataset_ids.to(device)
            with amp.autocast(enabled=True):
                with torch.no_grad():
                    feat = teacher(img, target, cam_label=target_cam, view_label=target_view, dataset_label=dataset_ids,
                                   forward_aux=False)
                stu_feat, stu_fc_feat = student(img, target, cam_label=target_cam, view_label=target_view,
                                                dataset_label=-1, forward_aux=False)

                if cfg.MODEL.AUX_LOSS:  # with consistent loss
                    if cfg.SOLVER.AUX_LOSS_TYPE == 'KL':
                        aux_loss = F.kl_div(F.log_softmax(stu_feat, dim=-1), F.log_softmax(feat, dim=-1),
                                            reduction='sum', log_target=True) / feat.size(0)
                    elif cfg.SOLVER.AUX_LOSS_TYPE == 'RKD':
                        with torch.no_grad():
                            t_d = pdist(feat, squared=False)
                            mean_td = t_d[t_d > 0].mean()
                            t_d = t_d / mean_td
                        d = pdist(stu_feat, squared=False)
                        mean_d = d[d > 0].mean()
                        d = d / mean_d
                        # RKD-D
                        loss_d = F.smooth_l1_loss(d, t_d)

                        with torch.no_grad():
                            td = (feat.unsqueeze(0) - feat.unsqueeze(1))
                            norm_td = F.normalize(td, p=2, dim=2)
                            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

                        sd = (stu_feat.unsqueeze(0) - stu_feat.unsqueeze(1))
                        norm_sd = F.normalize(sd, p=2, dim=2)
                        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
                        # RKD-A
                        loss_a = F.smooth_l1_loss(s_angle, t_angle)
                        aux_loss = loss_d + 2 * loss_a
                    else:
                        aux_loss = aux_loss_fn(stu_feat, feat)
                else:
                    aux_loss = torch.tensor(0.0).cuda()
                    stu_feat, stu_fc_feat = student(img, target, cam_label=target_cam, view_label=target_view,
                                                    dataset_label=dataset_ids, forward_aux=False)

                reid_loss = loss_fn(stu_fc_feat, stu_feat, target, target_cam)

            scaler.scale(reid_loss + cfg.SOLVER.AUX_LOSS_WEIGHT * aux_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (stu_fc_feat.max(1)[1] == target).float().mean()

            loss_meter.update(reid_loss.item(), img.shape[0])
            aux_loss_meter.update(cfg.SOLVER.AUX_LOSS_WEIGHT * aux_loss.item(), img.shape[0])
            acc_meter.update(acc, 1.0)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Reid Loss: {:.3f}, "
                    "AUX_LOSS: {:.3f}, "
                    "Reid Acc: {:.3f}, "
                    "Base Lr: {:.2e}".format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg,
                                             aux_loss_meter.avg,
                                             acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_epoch = end_time - start_time
        logger.info("Epoch {} done. Time per epoch: {:.3f}[s]".format(epoch, time_per_epoch))

        if epoch % checkpoint_period == 0:
            torch.save(student.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            student.eval()
            logger.info("------------------------------------")
            if cfg.MODEL.AUX_LOSS:
                print('test with aux prompts...')
                for name, evaluator in evaluator_dict.items():
                    evaluator.reset()
                for dataset_id, (name, val_loader) in enumerate(val_loader_dict.items()):
                    logger.info('Evaluating Dataset ' + name)
                    for n_iter, (img, pid, camid, camids, vid, target_view, imgpath) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = student(img, cam_label=camids, view_label=target_view, dataset_label=-1)
                            evaluator_dict[name].update((feat, pid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator_dict[name].compute()
                    logger.info("Validation Results for " + name + " - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    logger.info("------------------------------------")


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res
