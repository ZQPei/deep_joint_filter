import os
import torch

from src.models import DeepJointFilter


if __name__ == "__main__":
    net  = DeepJointFilter()
    target_image = torch.randn(10,1,32,32)
    guide_image = torch.randn(10,3,32,32)
    output = net(target_image, guide_image)

    import ipdb; ipdb.set_trace()