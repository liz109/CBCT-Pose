"""
Zero123 train for pose conditioned projection-generation
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
import torch
import torch.utils.checkpoint
import random
from datetime import datetime

from diffusers.utils import check_min_version

from src.utils import dict_to_namespace
# from solver import Solver
from src.train import Trainer


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")



DATASET_NAME_MAPPING = {
    "cbct-projections":("projections", "angles")
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config", type=str, default='./config/args.yaml', help="The argument yaml file."
    )

    opts = parser.parse_args()
    opts_dict = vars(opts)

    with open(opts.config, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)  
    args.update(opts_dict)
    args['device'] = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')


    args = dict_to_namespace(args)

    # args, unknown = parser.parse_known_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")


    return args




def main():
    args = parse_args()

    torch.cuda.empty_cache() 

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)



    solver = Trainer(args)
    solver.train()


if __name__ == '__main__':
    main()


""" 
RUN: change parameters in naf.yaml

$ python main.py --config config/naf.yaml

"""