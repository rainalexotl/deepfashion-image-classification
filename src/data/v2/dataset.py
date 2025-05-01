import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = 'data/raw'
MAX_SAMPLES_PER_CLASS = 320

class DeepFashionDataset(Dataset):
    def __init__(self, split='train', data_root=DATA_ROOT, transform=None, 
                 class_filename='list_category_cloth.txt',
                 downsample=False, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
        with open(os.path.join(data_root, f'{split}.txt')) as f:
            self.img_paths = [line.strip() for line in f]
        with open(os.path.join(data_root, f'{split}_cate.txt')) as f:
            self.labels = [int(line.strip()) for line in f]
        with open(os.path.join(data_root, f'{split}_bbox.txt')) as f:
            self.bboxes = [self._parse_coords(line.strip().split(' ')) for line in f]
        
        # Apply downsampling if requested
        if downsample:
            indices = self._downsample_by_class_metadata(self.labels, max_samples_per_class)
            self.img_paths = [self.img_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.bboxes = [self.bboxes[i] for i in indices]
        
        with open(os.path.join(ROOT, DATA_ROOT, class_filename)) as f:
            lines = f.readlines()[2:]  # skip first 2 lines
            self.classes = []

            for line in lines:
                category = line.strip().split()[0]  # get first word
                self.classes.append(category)

        self.data_root = data_root
        self.transform = transform

    def _downsample_by_class_metadata(self, labels, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
        from collections import defaultdict
        import random

        random.seed(2)

        class_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        selected_indices = []
        for label, indices in class_to_indices.items():
            if len(indices) > max_samples_per_class:
                indices = random.sample(indices, max_samples_per_class)
            selected_indices.extend(indices)

        return selected_indices

    def _parse_coords(self, bbox):
        """
        in: ['066', '075', '241', '293']
        out: [(66, 74), (241, 293)]
        """
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_root, self.img_paths[idx]))
        label = self.labels[idx]
        bbox = self.bboxes[idx]
        img = img.crop(bbox)
        if self.transform:
            img = self.transform(img)
        return img, label
    
def get_transforms(config):
    train_transform = T.Compose([
        T.Resize(config['data']['img_size']),
        T.RandomCrop(config['data']['img_size']),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1,0.1)),
        T.RandomRotation(20),
        T.GaussianBlur(kernel_size=3),
        T.ToTensor()
    ])

    val_transform = T.Compose([
        T.Resize(config['data']['img_size']),
        T.CenterCrop(config['data']['img_size']),
        T.ToTensor()
    ])

    if (config['model']['name'] == 'alexnet') or (config['model']['name'] == 'resnet34'):
        train_transform = T.Compose(train_transform.transforms + 
                              [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                              )
        
        val_transform = T.Compose(val_transform.transforms + 
                              [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                              )

    return train_transform, val_transform

def build_weighted_sampler(dataset, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    class_counts = Counter(dataset.labels)
    class_sample_counts = np.array([class_counts[i] for i in sorted(class_counts)])
    weights = 1. / class_sample_counts
    sample_weights = np.array([weights[sorted(class_counts).index(label)] for label in dataset.labels])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(class_counts) * max_samples_per_class,
        replacement=True
    )

    return sampler

def get_dataloaders(config, train=True):
    """
    Args:
        train (boolean): 

    Returns:
        train and val dataloaders if train=True, test dataloader if train=False
    """
    train_transform, val_transform = get_transforms(config)

    try:
        if train:
            train_dataset = DeepFashionDataset('train', transform=train_transform, downsample=True)
            sampler = build_weighted_sampler(train_dataset)
            val_dataset = DeepFashionDataset('val', transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)
        
            return train_loader, val_loader
        else:
            test_dataset = DeepFashionDataset('test', transform=val_transform)
            test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

            return test_loader
    except Exception as e:
        print(e)
        return