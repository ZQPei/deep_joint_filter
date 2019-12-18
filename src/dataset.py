import os
import random
import torch
import torchvision
import torchvision.transforms.functional as F
import PIL.Image as Image
import numpy as np
from torch.utils.data import DataLoader
from scipy.misc import imread, imresize

from .utils import load_flist


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        super(Dataset, self).__init__()

        self.config = config
        self.mode = mode

        self.flist_target = load_flist(config.dataset[mode].target)
        self.flist_guide = load_flist(config.dataset[mode].guide)
        self.flist_gt = load_flist(config.dataset[mode].gt)
        assert (len(self.flist_target) == len(self.flist_guide) and len(self.flist_target) == len(self.flist_gt))
        self.total = len(self.flist_target)

        self.input_size = config.dataset.input_size


    def __len__(self):
        return self.total


    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.load_name(index))
            item = self.load_item(0)

        return item


    def load_name(self, index):
        return os.path.basename(self.flist_gt[index])


    def load_item(self, index):
        fn_target = self.flist_target[index]
        fn_guide = self.flist_guide[index]
        fn_gt = self.flist_gt[index]

        # imread
        img_target = imread(fn_target)
        img_guide = imread(fn_guide)
        img_gt = imread(fn_gt)

        # transform
        tensor_target, tensor_guide, tensor_gt = self.transform(img_target, img_guide, img_gt)

        return tensor_target, tensor_guide, tensor_gt


    def transform(self, img_target, img_guide, img_gt):
        # to 3 channels
        img_target = self.check_channels(img_target)
        img_guide = self.check_channels(img_guide)
        img_gt = self.check_channels(img_gt)

        # random crop when training and center crop when testing
        if self.mode == "train":
            img_target, img_guide, img_gt = self.random_crop(img_target, img_guide, img_gt)
        else:
            img_target, img_guide, img_gt = self.center_crop(img_target, img_guide, img_gt)

        # resize
        img_target = self.resize(img_target)
        img_guide = self.resize(img_guide)
        img_gt = self.resize(img_gt)

        # add gaussian noise mannually
        if self.config.dataset.generate_noise:
            img_target = add_gaussian_noise(img_gt, self.config.dataset.noise_sigma)

        # to_tensor
        tensor_target = self.to_tensor(img_target)
        tensor_guide = self.to_tensor(img_guide)
        tensor_gt = self.to_tensor(img_gt)
        
        return tensor_target, tensor_guide, tensor_gt

    
    def check_channels(self, img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        return img


    def resize(self, img):
        return imresize(img, size=self.input_size)


    def random_crop(self, img_target, img_guide, img_gt):
        h, w, _ = img_target.shape
        tw, th = self.input_size

        i = random.randint(0, h - th if h - th > 0 else 0)
        j = random.randint(0, w - tw if w - tw > 0 else 0)

        return img_target[i:i+th, j:j+tw, :], img_guide[i:i+th, j:j+tw, :], img_gt[i:i+th, j:j+tw, :]


    def center_crop(self, img_target, img_guide, img_gt):
        h, w, _ = img_target.shape
        tw, th = self.input_size

        i = int(round((h - th) / 2.)) if h - th > 0 else 0
        j = int(round((w - tw) / 2.)) if w - tw > 0 else 0

        return img_target[i:i+th, j:j+tw, :], img_guide[i:i+th, j:j+tw, :], img_gt[i:i+th, j:j+tw, :]


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()  # scale of [0, 1]

        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )

            for item in sample_loader:
                yield item


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, target_folder, guide_folder):
        super(InferenceDataset, self).__init__()

        assert os.path.isdir(target_folder) and os.path.isdir(guide_folder)

        self.flist_target = load_flist(target_folder)
        self.flist_guide = load_flist(guide_folder)
        assert len(self.flist_target) == len(self.flist_guide)
        self.total = len(self.flist_target)



    def __len__(self):
        return self.total


    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.load_name(index))
            item = self.load_item(0)

        return item


    def load_name(self, index):
        return os.path.basename(self.flist_target[index])


    def load_item(self, index):
        fn_target = self.flist_target[index]
        fn_guide = self.flist_guide[index]

        # imread
        img_target = imread(fn_target)
        img_guide = imread(fn_guide)

        # transform
        tensor_target, tensor_guide = self.transform(img_target, img_guide)

        return tensor_target, tensor_guide


    def transform(self, img_target, img_guide):
        # to 3 channels
        img_target = self.check_channels(img_target)
        img_guide = self.check_channels(img_guide)


        # to_tensor
        tensor_target = self.to_tensor(img_target)
        tensor_guide = self.to_tensor(img_guide)
        
        return tensor_target, tensor_guide

    
    def check_channels(self, img):
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        return img


    def resize(self, img):
        return imresize(img, size=self.input_size)


    def random_crop(self, img_target, img_guide, img_gt):
        h, w, _ = img_target.shape
        tw, th = self.input_size

        i = random.randint(0, h - th if h - th > 0 else 0)
        j = random.randint(0, w - tw if w - tw > 0 else 0)

        return img_target[i:i+th, j:j+tw, :], img_guide[i:i+th, j:j+tw, :], img_gt[i:i+th, j:j+tw, :]


    def center_crop(self, img_target, img_guide, img_gt):
        h, w, _ = img_target.shape
        tw, th = self.input_size

        i = int(round((h - th) / 2.)) if h - th > 0 else 0
        j = int(round((w - tw) / 2.)) if w - tw > 0 else 0

        return img_target[i:i+th, j:j+tw, :], img_guide[i:i+th, j:j+tw, :], img_gt[i:i+th, j:j+tw, :]


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()  # scale of [0, 1]

        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            )

            for item in sample_loader:
                yield item
    

 
def add_gaussian_noise(image_in, noise_sigma=25):
    # image_in [0, 255]
    temp_image = np.float64(np.copy(image_in))
 
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    noisy_image.clip(0, 255)
    noisy_image = np.uint8(noisy_image)
    return noisy_image

