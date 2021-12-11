import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
import cv2

import scipy
import numpy as np

class ArcaneFaces(Dataset):
    def __init__(self, base_path, mode, transform=None, sz=320, rc=256):
        self.base_path = base_path
        self.mode = mode
        self.sz = sz
        self.rc = rc

        self.base_names = [f.split("/")[-1].replace(".jpg", "")
                           for f in glob.glob(os.path.join(self.base_path, 'faces', '*.jpg'))]
        print("len base names", len(self.base_names))
        self.images = [os.path.join(
            self.base_path, "faces", f"{f}.jpg") for f in self.base_names]
        self.masks = [os.path.join(
            self.base_path, "graycomics", f"{f}.jpg") for f in self.base_names]
        self.transform = transform

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):

        image = Image.open(self.images[idx])
        image.thumbnail((self.sz, self.sz), Image.ANTIALIAS)
        mask = Image.open(self.masks[idx])
        mask.thumbnail((self.sz, self.sz), Image.ANTIALIAS)

        if self.mode == "train":
            # if np.random.random() < 0.5:
            #     # hflip
            #     image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            #     mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            
            if np.random.random() < 0.5:
                # vflip
                image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)


        image = np.array(image).astype('float32')

        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]

        mask = np.array(mask).astype('float32')

        image /= 255.0
        image -= (0.485, 0.456, 0.406)
        image /= (0.229, 0.224, 0.225)
        mask /= 255.0

        mask = np.expand_dims(mask, -1)

        pad_x = int(self.sz - image.shape[0])
        pad_y = int(self.sz - image.shape[1])

        image = np.pad(image, ((0, pad_x), (0, pad_y), (0, 0)),
                       mode='constant')
        mask = np.pad(mask, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

        if self.mode == "train":
            # random crop to rc
            top = np.random.randint(0, self.sz - self.rc)
            left = np.random.randint(0, self.sz - self.rc)
            image = image[top: top + self.rc, left: left + self.rc]
            mask = mask[top: top + self.rc, left: left + self.rc]

        # if self.mode == "train":
        #     flipped_image = np.fliplr(image)
        #     flipped_mask = np.fliplr(mask)
        #     flipped_image = np.transpose(flipped_image, (2, 0, 1))
        #     flipped_mask = np.transpose(flipped_mask, (2, 0, 1))

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return torch.FloatTensor(image), torch.FloatTensor(mask)

        if self.mode == "train":
            return torch.FloatTensor(image), torch.FloatTensor(mask), torch.FloatTensor(flipped_image.copy()), torch.FloatTensor(flipped_mask.copy())
        else:
            return torch.FloatTensor(image), torch.FloatTensor(mask)