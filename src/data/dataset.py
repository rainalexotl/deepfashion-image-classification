import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as T

from src.data.sampler_utils import downsample_by_class, build_weighted_sampler

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = 'data/split'
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
    data = datasets.ImageFolder(os.path.join(ROOT, DATA_ROOT, split_type))
    classes = data.classes

    samples = downsample_by_class(data.imgs, max_samples_per_class)

    return CustomImageDataset(samples, classes, transform)

def get_transform(config):
    transform = T.Compose([
        T.Resize(config['data']['img_size']),
        T.CenterCrop(config['data']['img_size']),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1,0.1)),
        T.RandomRotation(20),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor()
    ])

    return transform

def get_dataloaders(config, train=True):
    """
    Args:
        train (boolean): 

    Returns:
        train and val dataloaders if train=True, test dataloader if train=False
    """
    
    try:
        if train:
            train_dataset = build_custom_dataset('train', transform=transform)
            sampler = build_weighted_sampler(train_dataset.samples)
            val_dataset = datasets.ImageFolder(os.path.join(ROOT, DATA_ROOT, 'val'), transform=transform)
            train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        
            return train_loader, val_loader
        else:
            test_dataset = datasets.ImageFolder(os.path.join(ROOT, DATA_ROOT, 'test'), transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

            return test_loader
    except Exception as e:
        print(e)
        return
