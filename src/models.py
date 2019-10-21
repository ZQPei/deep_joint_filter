import torch
import torch.nn as nn

from .networks import CNN


class DeepJointFilter(nn.Module):
    def __init__(self):
        super(DeepJointFilter, self).__init__()

        self.cnn_t = CNN(3, 1, [96,48,1], [9,1,5], [1,1,1], [2,2,2])
        self.cnn_g = CNN(3, 3, [96,48,1], [9,1,5], [1,1,1], [2,2,2])
        self.cnn_f = CNN(3, 2, [64,32,1], [9,1,5], [1,1,1], [0,0,0])


    def forward(self, target_image, guide_image):
        fmap1 = self.cnn_t(target_image)
        fmap2 = self.cnn_g(guide_image)
        output = self.cnn_f(torch.cat([fmap1, fmap2]))
        return output

    

