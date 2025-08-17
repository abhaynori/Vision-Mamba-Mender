import os
import PIL.Image as Image
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple, Optional


def _img_loader(path, mode='RGB'):
    assert mode in ['RGB', 'L']
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def _find_classes(root):
    class_names = [d.name for d in os.scandir(root) if d.is_dir()]
    class_names.sort()
    classes_indices = {class_names[i]: i for i in range(len(class_names))}
    # print(classes_indices)
    return class_names, classes_indices  # 'class_name':index


def _make_dataset(image_dir):
    samples = []  # image_path, class_idx

    class_names, class_indices = _find_classes(image_dir)

    for class_name in sorted(class_names):
        class_idx = class_indices[class_name]
        target_dir = os.path.join(image_dir, class_name)

        if not os.path.isdir(target_dir):
            continue

        for root, _, files in sorted(os.walk(target_dir)):
            for file in sorted(files):
                image_path = os.path.join(root, file)
                item = image_path, class_idx
                samples.append(item)
    return samples


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = _make_dataset(self.image_dir)
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = _img_loader(image_path, mode='RGB')
        name = os.path.split(image_path)[1]

        if self.transform is not None:
            image = self.transform(image)

        return image, target, name

    def __len__(self):
        return len(self.samples)


def create_synthetic_dataset(num_samples: int = 1000,
                           image_size: int = 32,
                           num_channels: int = 3,
                           num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic dataset for testing purposes
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of square images (image_size x image_size)
        num_channels: Number of channels (3 for RGB)
        num_classes: Number of classification classes
    
    Returns:
        Tuple of (images, labels)
    """
    
    # Generate random images with some structure
    images = torch.randn(num_samples, num_channels, image_size, image_size)
    
    # Add some patterns to make it more realistic
    for i in range(num_samples):
        # Add some geometric patterns based on class
        class_idx = i % num_classes
        
        if class_idx == 0:  # Horizontal lines
            images[i, :, ::4, :] += 2.0
        elif class_idx == 1:  # Vertical lines
            images[i, :, :, ::4] += 2.0
        elif class_idx == 2:  # Diagonal pattern
            for j in range(image_size):
                if j < image_size:
                    images[i, :, j, j] += 2.0
        elif class_idx == 3:  # Center bright
            center = image_size // 2
            images[i, :, center-2:center+2, center-2:center+2] += 3.0
        elif class_idx == 4:  # Corners bright
            images[i, :, :3, :3] += 2.0
            images[i, :, -3:, -3:] += 2.0
        # Other classes remain mostly random
    
    # Generate labels
    labels = torch.tensor([i % num_classes for i in range(num_samples)], dtype=torch.long)
    
    return images, labels


def create_vision_mamba_dataset(num_samples: int = 1000,
                               image_size: int = 32,
                               sequence_length: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dataset specifically for Vision Mamba testing
    
    Args:
        num_samples: Number of samples
        image_size: Size of images
        sequence_length: Length of sequence for Mamba processing
    
    Returns:
        Tuple of (sequences, labels)
    """
    
    # Create image sequences (simulating video or patch sequences)
    sequences = torch.randn(num_samples, sequence_length, 3, image_size, image_size)
    
    # Create labels based on sequence patterns
    labels = torch.randint(0, 10, (num_samples,))
    
    return sequences, labels
