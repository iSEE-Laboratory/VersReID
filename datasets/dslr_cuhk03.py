import os.path as osp
from .bases import BaseImageDataset
import os


class DSLR_CUHK03(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(DSLR_CUHK03, self).__init__()
        self.dataset_dir = root
        
        self.pid_begin = pid_begin
        train = self.load_data('train', relabel=True)
        query = self.load_data('query', relabel=False)
        gallery = self.load_data('gallery', relabel=False)

        if verbose:
            print("=> DSLR CUHK03 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        
    def load_data(self, split, relabel=False):
        lr_imgs, hr_imgs, pids, data_group = [], [], [], []
        data_dir = osp.join(self.dataset_dir, split)
        if split == 'query':
            resolutions = ['lr']
        elif split == 'gallery':
            resolutions = ['hr']
        else:
            resolutions = ['lr', 'hr']

        for resolution in resolutions:
            img_dir = osp.join(data_dir, resolution)
            for img_name in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, img_name)
                pid = int(img_name.split('_')[0])
                if pid not in pids:
                    pids.append(pid)
                cam = int(img_name.split('_')[2][0])
                if relabel:
                    pid = pids.index(pid)
                if resolution == 'lr':
                    lr_imgs.append((img_path, self.pid_begin + pid, cam, 1))
                else:
                    hr_imgs.append((img_path, self.pid_begin + pid, cam, 1))

        if split == 'train':
            data_group = hr_imgs + lr_imgs
        elif split == 'gallery':
            data_group = hr_imgs
        else:
            data_group = lr_imgs

        return data_group

