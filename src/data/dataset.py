import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from src.data.sampler_utils import downsample_by_class

ROOT = Path(__file__).resolve().parents[2]
MAX_SAMPLES_PER_CLASS = 320

class CustomImageDataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples
        self.img_path = [s[0] for s in samples]
        self.labels = [s[1] for s in samples]
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def build_custom_dataset(split_type, transform=None, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    data = datasets.ImageFolder(os.path.join(ROOT, split_type))
    classes = data.classes

    samples = downsample_by_class(data.imgs, MAX_SAMPLES_PER_CLASS)

    return CustomImageDataset(samples, classes, transform)
