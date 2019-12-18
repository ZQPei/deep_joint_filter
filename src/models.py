import torch
import torch.nn as nn
import torch.optim as optim
import os

from .networks import DeepJointFilterNetwork


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        self.iteration = 0
        self.weight_path = os.path.join(config.save_path, config.config_name, config.ckpt_path, config.config_name + ".pth")


    def load(self, weight_path=None):
        if os.path.isfile(self.weight_path):
            print("loading %s..."%(self.config.config_name))

            loaded_dict = torch.load(self.weight_path if weight_path is None else weight_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(loaded_dict['model'])
            self.iteration = loaded_dict['iteration']
        else:
            print("No checkpoints found!")


    def save(self, weight_path=None):
        print("saving %s..."%(self.config.config_name))

        torch.save({
            'iteration': self.iteration,
            'model': self.model.state_dict()
        }, self.weight_path if weight_path is None else weight_path)
        


class DeepJointFilterModel(BaseModel):
    def __init__(self, config):
        super(DeepJointFilterModel, self).__init__(config)

        self.config = config

        self.model = DeepJointFilterNetwork(config)

        if config.loss.name == "l2":
            self.loss = nn.MSELoss()
        elif config.loss.name == "l1":
            self.loss = nn.L1Loss()
        # self.optimizer = optim.SGD(
        #     params=self.model.parameters(),
        #     lr=float(config.trainer.lr),
        #     momentum=float(config.trainer.momentum)
        # )
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=float(config.trainer.lr),
            betas=(config.trainer.beta1, config.trainer.beta2)
        )


    def process(self, target_image, guide_image, gt=None):
        if self.training:
            self.iteration += 1

        if gt is not None:
            self.optimizer.zero_grad()

            output = self.model(target_image, guide_image)

            loss = 0
            mse_loss = self.loss(output, gt)

            loss += mse_loss
            logs = [("l_mse", mse_loss.item())]

            return output, loss, logs


        else:
            output = self.model(target_image, guide_image)

            return output



    def backward(self, loss):
        loss.backward()
        self.optimizer.step()



