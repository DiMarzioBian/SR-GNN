import os
import argparse
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class ASSD(Dataset):
    """ Event stream datasets. """
    def __init__(self,
                 list_filename: list,
                 mean,
                 std,
                 shrink_image,
                 augment_hvflip: float = 0,
                 augment_resize: float = 0,
                 root: str = '_data/assd/'):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        self._walker = list_filename
        self.length = len(self._walker)
        self.path_img = os.path.join(root, 'original_images')
        self.path_gt = os.path.join(root, 'label_images_semantic')
        self.ext_img = '.jpg'
        self.ext_gt = '.png'

        self.shrink_image = shrink_image
        self.augment_hvflip = augment_hvflip
        self.augment_resize = augment_resize

        self.to_tensor = T.Compose([
            T.ToTensor(),
        ])

        self.normalize = T.Compose([
            T.Normalize(mean=mean, std=std)
        ])

        self.shrink = T.Compose([
            T.Resize(self.shrink_image, T.InterpolationMode.NEAREST),
        ])

        self.h_flip = T.RandomHorizontalFlip(p=1.0)
        self.v_flip = T.RandomVerticalFlip(p=1.0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        path_img = self._walker[index] + self.ext_img
        path_gt = self._walker[index][:11] + 'label_images_semantic' + self._walker[index][-4:] + self.ext_gt

        # Preprocess
        img = self.to_tensor(np.array(Image.open(path_img)))
        gt = self.to_tensor(np.array(Image.open(path_gt))) * 255

        # Horizontal and vertical flip
        if np.random.rand() < self.augment_hvflip:
            p = np.random.rand()
            if p < 1/3:
                img = self.h_flip(img)
                gt = self.h_flip(gt)
            elif p < 2/3:
                img = self.v_flip(img)
                gt = self.v_flip(gt)
            else:
                img = self.h_flip(self.v_flip(img))
                gt = self.h_flip(self.v_flip(gt))

        # Random Crop
        if np.random.rand() < self.augment_resize:
            p = np.random.rand() / 2 + 0.5
            cropper = T.RandomCrop(size=(int(4000*p), int(6000*p)))

            img_gt = cropper(torch.cat((img, gt), 0))
            # To uniform output format
            img, gt = img_gt[:3, :, :], img_gt[3, :, :].unsqueeze(0)

        img = self.normalize(self.shrink(img))
        gt = self.shrink(gt.long()).squeeze(0)
        return img, gt


def get_assd_dataloader(opt: argparse.Namespace, train_list: list, val_list: list, test_list: list):
    """ Load data and prepare dataloader. """
    # Calculate mean and std
    if opt.recalculate_mean_std:
        mean, std = get_mean_std(train_list)
    else:
        mean = [0.44619015, 0.44376444, 0.40185362]
        std = [0.20309216, 0.19916435, 0.209552]

    # Instancelize datasets
    train_data = ASSD(list_filename=train_list, mean=mean, std=std, shrink_image=opt.shrink_image,
                      augment_hvflip=opt.enable_hvflip, augment_resize=opt.enable_resize)

    val_data = ASSD(list_filename=val_list, mean=mean, std=std, shrink_image=opt.shrink_image)

    test_data = ASSD(list_filename=test_list, mean=mean, std=std, shrink_image=opt.shrink_image)

    # Instancelize dataloader
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


def get_mean_std(train_list):
    """
    Calculate mean and std from given filename list
    """
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in tqdm(range(len(train_list)), desc='- (Calculating mean and std)   ', leave=False):
        fn = train_list[i]
        img = np.array(Image.open(fn + '.jpg'))
        for ch in range(img.shape[-1]):
            img_ch = img[:, :, ch]
            mean[ch] += img_ch.mean()
            std[ch] += img_ch.std()

    data_mean = mean / len(train_list)
    data_std = std / len(train_list)
    print('\n[info] Training data mean:', str(data_mean), 'and std:', str(data_std))
    return data_mean, data_std



