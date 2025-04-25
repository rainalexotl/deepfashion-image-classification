import os
from pathlib import Path

import random
import numpy as np
from collections import defaultdict, Counter

from torch.utils.data import WeightedRandomSampler

random.seed(2)

MAX_SAMPLES_PER_CLASS = 320

def downsample_by_class(data, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    # group samples by class
    class_to_samples = defaultdict(list)

    for path, label in data:
        class_to_samples[label].append((path, label))

    # downsample data
    downsampled_data = []
    for label, samples in class_to_samples.items():
        if len(samples) > 500:
            samples = random.sample(samples, max_samples_per_class)
        downsampled_data.extend(samples)

    return downsampled_data

def build_weighted_sampler(samples, max_samples_per_class=MAX_SAMPLES_PER_CLASS):
    labels = [label for _, label in samples]
    class_counts = Counter(labels) # dict -> class_counts[label] = count
    class_sample_count = np.array([class_counts[i] for i in range(len(class_counts))]) # get counts alone
    weights = 1. / class_sample_count
    # list of labels but with their weights instead ie. the probability of getting sampled:
    sample_weights = np.array([weights[label] for _, label in samples])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(class_counts) * max_samples_per_class,
        replacement=True
    )

    return sampler