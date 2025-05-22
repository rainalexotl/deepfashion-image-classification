import os
from pathlib import Path
import argparse
import yaml

import torch
import numpy as np
import random

from src.train.trainer import Trainer
from src.train.trainer import HFTrainer

ROOT = Path(__file__).resolve().parents[0]
CONFIG_ROOT = 'configs'

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to configuration file")
    parser.add_argument('-chk', '--checkpoint', type=str, help="Path to checkpoint file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    try:
        with open(os.path.join(ROOT, CONFIG_ROOT, args.config), 'r') as f:
            config = yaml.safe_load(f)
    except Exception as error:
        print(error)
        return
    
    if 'huggingface' in config['model'].get('source', ''):
        trainer = HFTrainer(config, checkpoint_path=args.checkpoint)
    elif 'timm' in config['model'].get('source', ''):
        trainer = None
    else:
        trainer = Trainer(config, checkpoint_path=args.checkpoint)

    try:
        trainer.train()
    except KeyboardInterrupt:
        checkpoint_path = os.path.join(config['experiment']['save_root'], config['experiment']['name'], 'checkpoints', 'last_model.pt')
        print(f"\nTraining Interrupted. Saving checkpoint to {checkpoint_path}")
        trainer.curr_epoch -= 1 # to restart back at interrupted epoch
        trainer.save_checkpoint(checkpoint_path)
        trainer.logger.save()
        print(f"Checkpoint saved to {checkpoint_path}. Exiting gracefully.")


if __name__ == '__main__':
    main()