import os
from pathlib import Path
import argparse
import yaml

from src.train.trainer import Trainer

ROOT = Path(__file__).resolve().parents[0]
CONFIG_ROOT = 'configs'

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
    
    trainer = Trainer(config, checkpoint_path=args.checkpoint)
    trainer.train()


if __name__ == '__main__':
    main()