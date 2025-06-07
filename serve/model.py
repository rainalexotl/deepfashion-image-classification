import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from transformers import SwinForImageClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.path.join(ROOT, 'experiments/swin/swin_best_model.pt')
# print(MODEL_PATH)

LABELS = ['Jumpsuit', 'Top', 'Bomber', 'Skirt', 'Tee', 'Sweatpants', 'Flannel', 
          'Tank', 'Turtleneck', 'Jersey', 'Blouse', 'Kaftan', 'Jeggings', 'Shirtdress', 
          'Sarong', 'Jacket', 'Leggings', 'Nightdress', 'Sweatshorts', 'Coverup']

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