import glob
import re
import os.path as osp
from .bases import BaseImageDataset
import os


class PRCC(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = root
        self.pid_begin = pid_begin

        self.cam_map = {'A': 0, 'B': 1, 'C': 2}

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.query_dir = osp.join(self.dataset_dir, 'test/A')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/C')

        train, train_id_list = self.init_dataset(self.train_dir)
        val, _ = self.init_dataset(self.val_dir)
        train = self.relabel_dataset(train, train_id_list)

        query, query_id_list = self.init_test_dataset(self.query_dir, is_query=True)
        gallery, gallery_id_list = self.init_test_dataset(self.gallery_dir, is_query=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = len(train_id_list)
        self.num_train_cams = 3
        self.num_train_vids = 1
        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, query, gallery)

    def relabel_dataset(self, dataset, id_list):
        new_dataset = []
        for filename, id, cam, index in dataset:
            new_dataset.append((filename, id_list.index(id) + self.pid_begin, cam, index))
        return new_dataset

    def _process_dir(self, dir_path, relabel=False, is_query=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        cam_container = set()
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        return dataset

    def init_dataset(self, root, must_in_ids=None):
        id_list = []
        dataset = []

        for _id in os.listdir(root):
            if int(_id) not in id_list:
                id_list.append(int(_id))
            if osp.isdir(osp.join(root, _id)):
                for filename in os.listdir(osp.join(root, _id)):
                    filename_list = filename.split('_')
                    cam = self.cam_map[filename_list[0]]
                    dataset.append((osp.join(root, _id, filename), int(_id), cam, 1))

        return dataset, id_list

    @staticmethod
    def init_test_dataset(root, is_query=False):
        id_list = []
        dataset = []
        per_cam = 0 if is_query else 1
        for _id in os.listdir(root):
            if int(_id) not in id_list:
                id_list.append(int(_id))
            if osp.isdir(osp.join(root, _id)):
                for filename in os.listdir(osp.join(root, _id)):
                    dataset.append((osp.join(root, _id, filename), int(_id), per_cam, 1))
        return dataset, id_list
