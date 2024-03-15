from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySamplerCombine(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            #  最后多余的不满num_instances的图不要了
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            # 随机选P个pid
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SceneBalanceSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances, len_train_list, num_scenes=4):
        self.data_source = data_source
        # len_train_list [45557, 38104, 15618, 34167] (general, cc, occ, ir)
        self.split_list = [0]
        for i in range(len(len_train_list)):
            self.split_list.append(self.split_list[-1] + len_train_list[i])
        self.batch_size = batch_size  # 64 or 128
        self.num_scenes = num_scenes  # 4
        self.num_instances = num_instances  # 4
        self.imgs_per_scenes = batch_size // num_scenes  # 64 // 4 = 16 or 128 // 4 = 32
        self.pids_per_scenes = self.imgs_per_scenes // num_instances  # 16 // 4 = 4 or 32 // 4 = 8
        print('Batch_size: {:d}, num_instances: {:d}, imgs_per_scenes: {:d}, pids_per_scenes: {:d}'.
              format(self.batch_size, self.num_instances, self.imgs_per_scenes, self.pids_per_scenes))

    def __iter__(self):
        scenes_idxs = []
        for i in range(self.num_scenes):
            if i == self.num_scenes - 1:
                scenes_idxs.append(RandomIdentitySamplerCombine(self.data_source[self.split_list[i]:],
                                                                self.imgs_per_scenes,
                                                                self.num_instances))
            else:
                scenes_idxs.append(
                    RandomIdentitySamplerCombine(self.data_source[self.split_list[i]: self.split_list[i + 1]],
                                                 self.imgs_per_scenes,
                                                 self.num_instances))

        # extend subset that is too small
        subset_len = []
        scenes_idxs_new = []
        for i, idxs in enumerate(scenes_idxs):
            curr_idxs = []
            for j in idxs:
                curr_idxs.append(j + self.split_list[i])
            subset_len.append(len(curr_idxs))
            scenes_idxs_new.append(curr_idxs)

        max_len = max(subset_len)
        power = []
        for length in subset_len:
            power.append(max_len // length + 1)

        for i in range(len(scenes_idxs_new)):
            scenes_idxs_new[i] = (scenes_idxs_new[i] * power[i])[:max_len]

        # combine the data from different subsets e.g., bs=128，then 0-31 for dataset 1，32-63 for dataset 2, ...
        final_idxs = []
        for i in range(len(scenes_idxs_new[0]) // self.imgs_per_scenes):
            for j in range(len(scenes_idxs_new)):
                final_idxs.extend(scenes_idxs_new[j][i * self.imgs_per_scenes: (i + 1) * self.imgs_per_scenes])

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
