import os
import shutil
import argparse
import torch

from src.parser import YamlParser
from src.deep_joint_filter import DeepJointFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./configs/djf_rgbnir256_gaussion25.yaml", help="config file")
    parser.add_argument("--target_dir", type=str, default="/data/pzq/RGB-NIR/tiff2png/train/noise")
    parser.add_argument("--guide_dir", type=str, default="/data/pzq/RGB-NIR/tiff2png/train/nir")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = YamlParser(config_file=args.config_file)

    djf  = DeepJointFilter(config)
    djf.load()
    djf.inference(args.target_dir, args.guide_dir, args.output_dir)

    import ipdb; ipdb.set_trace()