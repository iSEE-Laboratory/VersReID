# encoding: utf-8
"""
@author:  Drinky Yan
@contact: yanjk3@mail2.sysu.edu.cn
"""
from .bases import BaseImageDataset
from .market1501 import Market1501
from .occluded_duke import OccDukeMTMCreID
from .prcc import PRCC
from .msmt17 import MSMT17
from .celebrity import Celebrity
from .sysu_mm01 import SysuMM01
from .dslr_cuhk03 import DSLR_CUHK03

_CombineDataset__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'prcc': PRCC,
    'occluded_duke': OccDukeMTMCreID,
    'celebrity': Celebrity,
    'sysumm01': SysuMM01,
    'dslr_cuhk03': DSLR_CUHK03
}


class CombineDataset(BaseImageDataset):
    def __init__(self, names, roots, subset_type, combine_pid=True):
        super(CombineDataset, self).__init__()

        self.train_dict, train_statistic, train_statistic_scene = {}, {}, {}
        self.total_classes, self.len_train_subset, self.classes_train_subset = 0, [], []
        self.query_dict, query_statistic = {}, {}
        self.len_query_list = []
        self.gallery_dict, gallery_statistic = {}, {}

        num_scenes = len(set(subset_type))
        pid_begin_per_subset = [0] * num_scenes
        pid_begin = 0
        for i, dataset_name in enumerate(names):
            print(f'Loading dataset {dataset_name}, path {roots[i]}')
            if combine_pid:
                dataset = _CombineDataset__factory[dataset_name](root=roots[i],
                                                                 pid_begin=pid_begin,
                                                                 verbose=False)
            else:
                dataset = _CombineDataset__factory[dataset_name](root=roots[i],
                                                                 pid_begin=pid_begin_per_subset[subset_type[i]],
                                                                 verbose=False)

            self.train_dict[dataset_name] = dataset.train
            # statistic of single training subset
            train_statistic[dataset_name] = self.get_imagedata_info(dataset.train)

            # statistic of total classes
            self.total_classes += train_statistic[dataset_name][0]

            # statistic of single training scene
            if subset_type[i] not in train_statistic_scene.keys():
                train_statistic_scene[subset_type[i]] = list(self.get_imagedata_info(dataset.train))
            else:
                new_statistic = list(self.get_imagedata_info(dataset.train))
                for j in range(len(train_statistic_scene[subset_type[i]])):
                    train_statistic_scene[subset_type[i]][j] += new_statistic[j]

            self.query_dict[dataset_name] = dataset.query
            query_statistic[dataset_name] = self.get_imagedata_info(dataset.query)

            self.gallery_dict[dataset_name] = dataset.gallery
            gallery_statistic[dataset_name] = self.get_imagedata_info(dataset.gallery)

            if combine_pid:
                pid_begin += self.get_imagedata_info(dataset.train)[0]
            else:
                pid_begin_per_subset[subset_type[i]] += self.get_imagedata_info(dataset.train)[0]

        self.train = []
        # reorganize the training data, append which type of subset this image is
        for i, (name, train_dataset) in enumerate(self.train_dict.items()):
            new_train_data = []
            for data in train_dataset:
                img_path, p_id, cam_id, view_id = data
                new_train_data.append(
                    (img_path, p_id, cam_id, view_id, subset_type[i]))  # i is the index of the dataset
            self.train += new_train_data

        print("--------------------------------------------")
        print('The combine dataset contains:')
        print(names)
        print("--------------------------------------------")
        print('Training set contains:')
        print("subset name   | # ids | # images | # cameras")
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(train_statistic[name][0], train_statistic[name][1],
                                                 train_statistic[name][2]))

        for i in range(num_scenes):
            self.classes_train_subset.append(train_statistic_scene[i][0])
            self.len_train_subset.append(train_statistic_scene[i][1])

        if combine_pid:
            total_train_statistic = self.get_dataset_info(self.train)
            print("--------------------------------------------")
            print('The total training set contains:')
            print("# ids | # images | # cameras  | # scene")
            print('{:5d} | {:8d} | Don\'t care | {:8d}'.format(total_train_statistic[0], total_train_statistic[1],
                                                               total_train_statistic[3]))
        else:
            print('W/O combining pid, the training subsets are separate.')
            print("subset type   | # ids | # images | # cameras")
            for i in range(num_scenes):
                print((str(i) + 13 * ' ' + '| '), end='')
                print('{:5d} | {:8d} | Don\'t care'.format(train_statistic_scene[i][0], train_statistic_scene[i][1]))
        print("--------------------------------------------")

        print('Query set contains:')
        print("subset        | # ids | # images | # cameras")
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(query_statistic[name][0], query_statistic[name][1],
                                                 query_statistic[name][2]))
            self.len_query_list.append(query_statistic[name][1])
        print("--------------------------------------------")

        print('Gallery set contains:')
        print("subset        | # ids | # images | # cameras")
        for name in names:
            print((name + (14 - len(name)) * ' ' + '| '), end='')
            print('{:5d} | {:8d} | {:9d}'.format(gallery_statistic[name][0], gallery_statistic[name][1],
                                                 gallery_statistic[name][2]))
        print("--------------------------------------------")

    @staticmethod
    def get_dataset_info(data):
        pids, cams, vids, dataset_ids = [], [], [], []

        for _, pid, camid, vid, dataset_id in data:
            pids += [pid]
            cams += [camid]
            vids += [vid]
            dataset_ids += [dataset_id]
        pids = set(pids)
        cams = set(cams)
        vids = set(vids)
        dataset_ids = set(dataset_ids)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(vids)
        num_datasets = len(dataset_ids)
        return num_pids, num_imgs, num_cams, num_datasets
