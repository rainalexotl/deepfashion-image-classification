import os
from pathlib import Path
from collections import Counter
from collections import defaultdict
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T
        

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = 'data/raw'
MAX_SAMPLES_PER_CLASS = 500

class DeepFashionDataset(Dataset):
    def __init__(self, split='train', data_root=DATA_ROOT, transform=None, 
                 class_filename='list_category_cloth.txt',
                 downsample=False, max_samples_per_class=MAX_SAMPLES_PER_CLASS,
                 include_labels=None):
        with open(os.path.join(ROOT, data_root, f'{split}.txt')) as f:
            self.img_paths = [line.strip() for line in f]
        with open(os.path.join(ROOT, data_root, f'{split}_cate.txt')) as f:
            self.labels = [int(line.strip()) for line in f]
        with open(os.path.join(ROOT, data_root, f'{split}_bbox.txt')) as f:
            self.bboxes = [self._parse_coords(line.strip().split(' ')) for line in f]
        
        with open(os.path.join(ROOT, DATA_ROOT, class_filename)) as f:
            lines = f.readlines()[2:]  # skip first 2 lines
            self.classes = []

            for line in lines:
                category = line.strip().split()[0]  # get first word
                self.classes.append(category)

        if include_labels:
            self._filter_data(include_labels)
        
        # Apply downsampling if requested
        if downsample:
            indices = self._downsample_by_class_metadata(self.labels, max_samples_per_class)
            self.img_paths = [self.img_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.bboxes = [self.bboxes[i] for i in indices]

        self.data_root = os.path.join(ROOT, DATA_ROOT)
        self.transform = transform

    def _downsample_by_class_metadata(self, labels, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
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
    
    def _filter_data(self, include_labels):
        # Create a mapping from original label index → new index (0 to len(include_labels)-1)
        label_map = {label: idx for idx, label in enumerate(include_labels)}

        filtered = []
        new_labels = []

        for x, y, box in zip(self.img_paths, self.labels, self.bboxes):
            if self.classes[y] in include_labels:
                new_labels.append(label_map[self.classes[y]])
                filtered.append((x, box))

        if filtered:
            self.img_paths, self.bboxes = zip(*filtered)
            self.labels = new_labels
        else:
            self.img_paths, self.labels, self.bboxes = [], [], []

        # Update self.classes to match the filtered set (ordered by include_labels)
        self.classes = include_labels

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
    
"""
Dataset wrapper for huggingface models
"""
class HFDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = dataset.classes
        self.labels = dataset.labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {
            "pixel_values": image,
            "labels": label
        }
    
def get_transforms(config, processor=None):
    train_transform_steps = [T.ToImage()]

    # Resize and crop
    if processor:
        if config['model'].get('source', '') == 'timm':
            _, height, width = processor['input_size']
            size = (height, width)
        elif config['model'].get('source', '') == 'huggingface':
            size = (processor.size['height'], processor.size['width'])
        else: 
            raise ValueError(f"Unsupported model source: {config['model']['source']}")
    else:
        size = config['data']['img_size']

    train_transform_steps.extend([
        T.Resize(size),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
        T.RandomRotation(15),
        T.GaussianBlur(kernel_size=3),
        T.ToDtype(torch.float32, scale=True)
    ])

    # Only normalize if processor is provided
    if processor:
        if config['model'].get('source', '') == 'timm':
            mean = processor['mean']
            std = processor['std']
        elif config['model'].get('source', '') == 'huggingface':
            mean = processor.image_mean
            std = processor.image_std
        train_transform_steps.append(T.Normalize(mean=mean, std=std))
    elif config['model']['name'] in ['alexnet', 'resnet34']:
        train_transform_steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = T.Compose(train_transform_steps)

    # -------

    val_transform_steps = [T.ToImage()]

    if processor:
        if config['model'].get('source', '') == 'timm':
            _, height, width = processor['input_size']
            size = (height, width)
        elif config['model'].get('source', '') == 'huggingface':
            size = (processor.size['height'], processor.size['width'])
    else:
        size = config['data']['img_size']

    val_transform_steps.extend([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToDtype(torch.float32, scale=True),
    ])

    if processor:
        if config['model'].get('source', '') == 'timm':
            mean = processor['mean']
            std = processor['std']
        elif config['model'].get('source', '') == 'huggingface':
            mean = processor.image_mean
            std = processor.image_std
        val_transform_steps.append(T.Normalize(mean=mean, std=std))
    elif config['model']['name'] in ['alexnet', 'resnet34']:
        val_transform_steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    val_transform = T.Compose(val_transform_steps)

    return train_transform, val_transform

def build_weighted_sampler(dataset, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    label_counts = Counter(dataset.labels)
    label_weights = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [label_weights[label] for label in dataset.labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(label_counts) * max_samples_per_class,
        replacement=True
    )

    return sampler

def get_datasets(config, processor=None, train=True):
    
    train_transform, val_transform = get_transforms(config, processor)
    downsample = config['data'].get('downsample', False)
    max_samples = config['data'].get('max_samples', MAX_SAMPLES_PER_CLASS)
    include_labels = config['data'].get('include_labels', None)
    
    try:
        if train:
            train_dataset = DeepFashionDataset('train', transform=train_transform, downsample=downsample, 
                                               max_samples_per_class=max_samples, include_labels=include_labels)
            val_dataset = DeepFashionDataset('val', transform=val_transform, include_labels=include_labels)

            if config['model'].get('source', '') == 'huggingface':
                train_dataset = HFDatasetWrapper(train_dataset)
                val_dataset = HFDatasetWrapper(val_dataset)

            return train_dataset, val_dataset
        else:
            test_dataset = DeepFashionDataset('test', transform=val_transform, include_labels=include_labels)

            if config['model'].get('source', '') == 'huggingface':
                test_dataset = HFDatasetWrapper(test_dataset)

            return test_dataset
        
    except Exception as e:
        print("Error getting datasets")
        raise e


def get_dataloaders(config, processor=None, train=True):
    """
    Args:
        train (boolean): 

    Returns:
        train and val dataloaders if train=True, test dataloader if train=False
    """

    try:
        if train:
            train_dataset, val_dataset = get_datasets(config, processor, train)
            sampler = build_weighted_sampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=False)

            return train_loader, val_loader
        else:
            test_dataset = get_datasets(config, processor, train)
            test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

            return test_loader
    except Exception as e:
        print("Error getting dataloaders")
        raise e