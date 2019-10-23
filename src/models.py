import torch
import torch.nn as nn
import os

from .networks import CNN


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.iteration = 0

        self.weight_path = os.path.join(config.config_path, name + ".pth")

    
    def load(self):
        if os.path.exists(self.weight_path):
            print('Loading %s ...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.weight_path)
            else:
                data = torch.load(self.weight_path, map_location=lambda storage, loc: storage)

            self.iteration = data['iteration']

    
    def save(self):
        print('Saving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'encoder': self.encoder.state_dict(),
        }, self.weight_path)
        


class DeepJointFilter(BaseModel):
    def __init__(self):
        super(DeepJointFilter, self).__init__()

        self.cnn_t = CNN(3, 1, [96,48,1], [9,1,5], [1,1,1], [2,2,2])
        self.cnn_g = CNN(3, 3, [96,48,1], [9,1,5], [1,1,1], [2,2,2])
        self.cnn_f = CNN(3, 2, [64,32,1], [9,1,5], [1,1,1], [0,0,0])


    def forward(self, target_image, guide_image):
        fmap1 = self.cnn_t(target_image)
        fmap2 = self.cnn_g(guide_image)
        output = self.cnn_f(torch.cat([fmap1, fmap2], dim=1))
        return output

    

