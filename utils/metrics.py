import torch
import numpy as np
from utils.reranking import re_ranking
import random


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
        Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, dataset=None, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.dataset = dataset

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        if self.dataset == 'prcc':
            cmc, mAP, _ = eval_prcc(distmat, q_pids, g_pids, q_camids, g_camids)
        elif self.dataset == 'sysumm01':
            cmc, mAP, _ = eval_sysumm01(distmat, q_pids, g_pids, q_camids, g_camids)
        else:
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf


def eval_prcc(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    cmc_sum = 0.0
    map_sum = 0.0

    id2idx = {}
    for i in range(len(g_pids)):
        if id2idx.get(g_pids[i]) is None:
            id2idx[g_pids[i]] = []
        id2idx[g_pids[i]].append(i)

    for i in range(10):
        random.seed(i)
        tmp = []  # temporary
        for key in id2idx:
            tmp.append(random.choice(id2idx[key]))
        assert len(tmp) == 71

        tmpg_pids = g_pids[tmp]
        tmpg_camids = g_camids[tmp]
        tmp_distmat = distmat[:, tmp]

        cmc, mAP, _ = eval_prcc_one_trial(tmp_distmat, q_pids, tmpg_pids, q_camids, tmpg_camids, max_rank)
        cmc_sum += cmc
        map_sum += np.mean(mAP)
        print('Trial {} | Rank-1: {:.2%} | mAP: {:.2%}'.format(i + 1, cmc[0], np.mean(mAP)))

    all_AP = map_sum / 10.0
    all_cmc = cmc_sum / 10.0

    return all_cmc, all_AP, 0.


def eval_sysumm01(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    cmc_sum = 0.0
    map_sum = 0.0

    id2idx = {}
    for i in range(len(g_pids)):
        if id2idx.get(g_pids[i]) is None:
            id2idx[g_pids[i]] = {}
        if id2idx[g_pids[i]].get(g_camids[i]) is None:
            id2idx[g_pids[i]][g_camids[i]] = []
        id2idx[g_pids[i]][g_camids[i]].append(i)

    for i in range(10):
        random.seed(i)
        tmp = []  # temporary
        for p_id in id2idx:
            for c_id in id2idx[p_id]:
                tmp.append(random.choice(id2idx[p_id][c_id]))

        tmpg_pids = g_pids[tmp]
        tmpg_camids = g_camids[tmp]
        tmp_distmat = distmat[:, tmp]

        cmc, mAP, _ = eval_sysu_mm01_one_trial(tmp_distmat, q_pids, tmpg_pids, q_camids, tmpg_camids, max_rank)
        cmc_sum += cmc
        map_sum += np.mean(mAP)
        print('Trial {} | Rank-1: {:.2%} | mAP: {:.2%}'.format(i + 1, cmc[0], np.mean(mAP)))

    all_AP = map_sum / 10.0
    all_cmc = cmc_sum / 10.0

    return all_cmc, all_AP, 0.


def eval_prcc_one_trial(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def eval_sysu_mm01_one_trial(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP