import os
import os.path as osp


def process_path(path):
    return osp.abspath(osp.expanduser(path))