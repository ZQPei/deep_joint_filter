import os
import shutil
import torch

from src.parser import YamlParser
from src.deep_joint_filter import DeepJointFilterModel


if __name__ == "__main__":
    config = YamlParser(config_file="./config.yml")

    deep_joint_filter  = DeepJointFilterModel(config)
    deep_joint_filter.load("./")
    deep_joint_filter.test()

    import ipdb; ipdb.set_trace()