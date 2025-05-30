import os
from pathlib import Path
import argparse
import yaml

import torch
import numpy as np
import random

from src.train.evaluate import evaluate, hf_evaluate

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = 'configs'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Configuration file")
    parser.add_argument('-chk', '--checkpoint', type=str, required=True, help="Path to checkpoint file to evaluate")
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    try:
        with open(os.path.join(ROOT, CONFIG_ROOT, args.config), 'r') as f:
            config = yaml.safe_load(f)
    except Exception as error:
        raise error

    try:
        if 'huggingface' in config['model'].get('source', ''):
            hf_evaluate(config, args.checkpoint, save_results=args.save)

        else:
            evaluate(config, args.checkpoint, save_results=args.save)
    except Exception as error:
        raise error

if __name__ == '__main__':
    main()