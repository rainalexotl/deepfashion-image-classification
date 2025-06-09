import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from transformers import SwinForImageClassification
from huggingface_hub import hf_hub_download

LABELS = ['Jumpsuit', 'Top', 'Bomber', 'Skirt', 'Tee', 'Sweatpants', 'Flannel', 
          'Tank', 'Turtleneck', 'Jersey', 'Blouse', 'Kaftan', 'Jeggings', 'Shirtdress', 
          'Sarong', 'Jacket', 'Leggings', 'Nightdress', 'Sweatshorts', 'Coverup']

REPO_ID = 'rainalexotl/deepfashion-image-classification-swin'
FILENAME = 'swin_best_model.pt'
HF_TOKEN = os.getenv('HF_TOKEN') 
MODEL_PATH = Path(
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN
    )
)

def load_model():
    model = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224', 
                                                       num_labels=len(LABELS), ignore_mismatched_sizes=True)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def make_prediction(model, image):
    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)

    with torch.no_grad():
        outputs = model(pixel_values=image.unsqueeze(0))
    prediction = outputs.logits.argmax(dim=1).item()
    return LABELS[prediction]