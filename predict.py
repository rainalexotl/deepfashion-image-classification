import os
from pathlib import Path
import argparse
import yaml

from PIL import Image
import torch

from src.data.v2.dataset import get_transforms
from src.models.factory import get_model

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = 'configs'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inference_file', help="Path to image file to make prediction on")
    parser.add_argument('-c', '--config', type=str, required=True, help="Configuration file")
    parser.add_argument('-chk', '--checkpoint', type=str, required=True, help="Path to checkpoint file to evaluate")
    parser.add_argument('--topk', type=int)
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
    
    try:
        img = Image.open(args.inference_file).convert('RGB')
    except Exception as error:
        print(error)
        return

    _, transform = get_transforms(config)
    img = transform(img)

    model = get_model(config)
    checkpoint = torch.load(os.path.join(ROOT, args.checkpoint))
    model.load_state_dict(checkpoint['model_state_dict'])

    
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0)) # add batch dimension

    print(f"Predicted class: {pred.argmax(dim=1).item()}")
    if args.topk:
        _, top_idx = torch.topk(pred, args.topk)
        print(f"Top {args.topk}: {top_idx.tolist()[0]}")


if __name__ == '__main__':
    main()