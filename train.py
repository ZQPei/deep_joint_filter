import os
import torch

from src.parser import YamlParser
from src.deep_joint_filter import DeepJointFilter


if __name__ == "__main__":
    config = YamlParser(config_path="./config.yml")
    # import ipdb; ipdb.set_trace()
    deep_joint_filter  = DeepJointFilter(config)
    deep_joint_filter.train()

    import ipdb; ipdb.set_trace()