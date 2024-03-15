import os.path as osp
import os
from .bases import BaseImageDataset


class Celebrity(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Celebrity, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, False)
        query = self._process_dir(self.query_dir, True)
        gallery = self._process_dir(self.gallery_dir, False)

        if verbose:
            print("=> Celebrity loaded")
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_query):
        dataset = []
        train_set = os.listdir(dir_path)
        cam_id = 0 if is_query else 1
        for img_name in train_set:
            pid = int(img_name.split('_')[0]) - 1
            img_path = osp.join(dir_path, img_name)
            dataset.append((img_path, self.pid_begin + pid, cam_id, 1))
        return dataset
