import torch
import torch.nn as nn
import torch.optim as optim
import os

from .networks import DeepJointFilter


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        self.iteration = 0
        self.weight_path = os.path.join(config.config_path, config.name + ".pth")

    def load(self):
        print("loading %s..."%(self.config.name))

        loaded_dict = torch.load(self.weight_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(loaded_dict['model'])
        self.iteration = loaded_dict['iteration']


    def save(self):
        print("saving %s..."%(self.config.name))

        torch.save({
            'iteration': self.iteration,
            'model': self.model.state_dict()
        }, self.weight_path)
        


class DeepJointFilterModel(BaseModel):
    def __init__(self, config):
        super(DeepJointFilterModel, self).__init__()

        self.config = config

        self.model = DeepJointFilter()

        self.l2_loss = nn.MSELoss()
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=float(config.trainer.lr),
            momentum=float(config.trainer.momentum)
        )


    def process(self, target_image, guide_image, gt):
        output = self.model(target_image, guide_image)

        loss = 0

        mse_loss = self.l2_loss(output, gt)
        
        loss += mse_loss
        loss.backward()
        self.optimizer.step()

        logs = ["l_mse", mse_loss.item()]

        return output, loss, logs







    

    

