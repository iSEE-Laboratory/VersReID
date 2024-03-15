import os
import os.path as osp
from .bases import BaseImageDataset


class SysuMM01(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(SysuMM01, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = root
        with open(osp.join(self.dataset_dir, 'exp/train_id.txt'), 'r') as f:
            for line in f:
                train_id = [int(i) for i in line.strip().split(',')]
        with open(osp.join(self.dataset_dir, 'exp/val_id.txt'), 'r') as f:
            for line in f:
                val_id = [int(i) for i in line.strip().split(',')]
        with open(osp.join(self.dataset_dir, 'exp/test_id.txt'), 'r') as f:
            for line in f:
                test_id = [int(i) for i in line.strip().split(',')]

        train = self._process_dir(train_id, range(1, 7))
        val = self._process_dir(val_id, range(1, 7))
        query = self._process_dir(test_id, [3, 6])
        gallery = self._process_dir(test_id, [1, 2, 4, 5])

        train = train + val
        train = self.relabel(train)

        if verbose:
            print("=> SYSUMM01 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _process_dir(self, p_ids, cam_ids, if_query=False):
        dataset = []
        for cam_id in cam_ids:
            cam_data = osp.join(self.dataset_dir, 'cam' + str(cam_id))
            pids_in_this_cam = os.listdir(cam_data)
            for pid in pids_in_this_cam:
                if int(pid) in p_ids:
                    data_root = osp.join(cam_data, pid)
                    for img in os.listdir(data_root):
                        img_path = osp.join(data_root, img)
                        if not if_query:
                            dataset.append((img_path, int(pid), cam_id - 1, 1))
                        else:
                            if cam_id == 3:
                                dataset.append((img_path, int(pid), 1, 1))
                            else:
                                dataset.append((img_path, int(pid), cam_id - 1, 1))
        return dataset

    def relabel(self, dataset):
        new_dataset = []
        id_dict = {}
        for data in dataset:
            img_path, p_id, cam_id, view_id = data
            if p_id not in id_dict:
                id_dict[p_id] = len(id_dict) + self.pid_begin
            new_dataset.append((img_path, id_dict[p_id], cam_id, view_id))
        return new_dataset
