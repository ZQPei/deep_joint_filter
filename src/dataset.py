import os
import torch
import torchvision
import torchvision.transforms.functional as F
import PIL.Image as Image
from torch.utils.data import DataLoader
from scipy.misc import imread, imresize

from .utils import load_flist


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        super(Dataset, self).__init__()

        self.config = config

        self.flist_target = load_flist(config.dataset.path.target)
        self.flist_guide = load_flist(config.dataset.path.guide)
        self.flist_gt = load_flist(config.dataset.path.gt)
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
        return os.path.basname(self.flist_gt[index])


    def load_item(self, index):
        fn_target = self.flist_target[index]
        fn_guide = self.flist_guide[index]
        fn_gt = self.flist_gt[index]

        img_target = imread(fn_target)
        img_guide = imread(fn_guide)
        img_gt = imread(fn_gt)

        img_target = self.transform(img_target)
        img_guide = self.transform(img_guide)
        img_gt = self.transform(img_gt)

        tensor_target = self.to_tensor(img_target)
        tensor_guide = self.to_tensor(img_guide)
        tensor_gt = self.to_tensor(img_gt)

        return tensor_target, tensor_guide, tensor_gt


    def transform(self, img):
        img = self.resize(img)

        return img


    def resize(self, img):
        return imresize(img, size=self.input_size)


    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()

        return img_t


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
    




