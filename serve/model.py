import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from huggingface_hub import hf_hub_download
import onnxruntime as ort

LABELS = ['Jumpsuit', 'Top', 'Bomber', 'Skirt', 'Tee', 'Sweatpants', 'Flannel', 
          'Tank', 'Turtleneck', 'Jersey', 'Blouse', 'Kaftan', 'Jeggings', 'Shirtdress', 
          'Sarong', 'Jacket', 'Leggings', 'Nightdress', 'Sweatshorts', 'Coverup']

REPO_ID = 'rainalexotl/deepfashion-image-classification-swin'
FILENAME = 'swin_model.onnx'
HF_TOKEN = os.getenv('HF_TOKEN') 
MODEL_PATH = Path(
    hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN
    )
)

def load_model():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def make_prediction(model, image):
    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224)),
        T.CenterCrop((224, 224)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0).numpy()

    outputs = model.run(None, {"pixel_values": image})
    logits = outputs[0]
    pred_idx = np.argmax(logits, axis=1)[0]
    return LABELS[pred_idx]