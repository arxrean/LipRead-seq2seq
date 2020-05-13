import os
import sys
import pdb
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms

sys.path.append('./')
import utils
from option import get_parser


def crop_video(opt):
    assert os.path.exists(opt.lrw_root)
    filenames = glob.glob(os.path.join(opt.lrw_root, '*', '*', '*.mp4'))
    for filename in tqdm(filenames):
        if len(filename.split('/')[-1].split('_')) == 2:
            data = utils.get_imgs_from_video(filename, RGB=True)[:, 115:211, 79:175]
            path_to_save = os.path.join(filename[:-4] + '_mouth.npz')
            np.savez(path_to_save, data=data)


class LRWClassification:
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args

        if mode == 'train':
            self.cur_data = glob.glob(os.path.join(args.lrw_root, '*', 'train', '*_mouth.npz'))
        elif mode == 'val':
            self.cur_data = glob.glob(os.path.join(args.lrw_root, '*', 'val', '*_mouth.npz'))
        else:
            self.cur_data = glob.glob(os.path.join(args.lrw_root, '*', 'test', '*_mouth.npz'))

        self.transform = self.get_transform()
        self.label_idx_list = list(np.load('./repo/lrw_label.npy'))

    def __getitem__(self, index):
        item = self.cur_data[index]

        video = np.load(item)['data']
        array = [Image.fromarray(f) for f in video]
        array = [self.transform(im) for im in array]
        array = torch.stack(array)

        label = item.split('/')[-3]
        label_idx = self.label_idx_list.index(label)

        return array, label_idx, label, item

    def __len__(self):
        return len(self.cur_data)

    def get_transform(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop((88, 88)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Grayscale(),
                transforms.TenCrop((88, 88)),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops])),
            ])


if __name__ == '__main__':
    opt = get_parser()
    crop_video(opt)
