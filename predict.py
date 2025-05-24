import os
from pathlib import Path
import argparse
import yaml

from PIL import Image
import torch
from transformers import ( ViTForImageClassification, ViTImageProcessor,
                          SwinForImageClassification, AutoImageProcessor )

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
        raise error
    
    img = Image.open(args.inference_file).convert('RGB')

    try:
        if 'huggingface' in config['model'].get('source', ''):
            if config['model']['type'] == 'vit':
                model = ViTForImageClassification.from_pretrained(args.checkpoint)
                processor = ViTImageProcessor.from_pretrained(config['model']['name'])
            elif config['model']['type'] == 'swin':
                model = SwinForImageClassification.from_pretrained(args.checkpoint)
                processor = AutoImageProcessor.from_pretrained(config['model']['name'])
            else:
                raise ValueError(f"Unsupported model type: {config['model']['type']}")
            
            _, transform = get_transforms(config, processor)
            img = transform(img)

            model.eval()
            with torch.no_grad():
                outputs = model(pixel_values=img.unsqueeze(0))
                pred = outputs.logits.argmax(dim=1).item()

            print(f"Predicted class: {pred}")
            
            final_pred = outputs.logits # for potential top5


        else:
            # img = Image.open(args.inference_file).convert('RGB')
            _, transform = get_transforms(config)
            img = transform(img)

            model = get_model(config)
            checkpoint = torch.load(os.path.join(ROOT, args.checkpoint))
            model.load_state_dict(checkpoint['model_state_dict'])

            model.eval()
            with torch.no_grad():
                pred = model(img.unsqueeze(0)) # add batch dimension
            
            print(f"Predicted class: {pred.argmax(dim=1).item()}")

            final_pred = pred # for potential top5

        if args.topk:
            _, top_idx = torch.topk(final_pred, args.topk)
            print(f"Top {args.topk}: {top_idx.tolist()[0]}")

    except Exception as error:
        raise error

if __name__ == '__main__':
    main()