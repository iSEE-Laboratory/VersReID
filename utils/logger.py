import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train, file_name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            file_name = 'train_log.txt' if file_name is None else file_name
            fh = logging.FileHandler(os.path.join(save_dir, file_name), mode='w')
        else:
            file_name = 'test_log.txt' if file_name is None else file_name
            fh = logging.FileHandler(os.path.join(save_dir, file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
