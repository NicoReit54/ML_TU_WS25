import os
import cv2

import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset
from torchvision.transforms import v2

IMG_SIZE = (64, 64) # decided to stay in line with base line configuration (sift needed it)

GTSRB_CLASSES = {
    0: 'Speed 20', 1: 'Speed 30', 2: 'Speed 50', 3: 'Speed 60', 4: 'Speed 70', 
    5: 'Speed 80', 6: 'End Speed 80', 7: 'Speed 100', 8: 'Speed 120', 9: 'No passing', 
    10: 'No passing >3.5t', 11: 'Right-of-way', 12: 'Priority road', 13: 'Yield', 
    14: 'Stop', 15: 'No vehicles', 16: 'Veh >3.5t prohib', 17: 'No entry', 
    18: 'General caution', 19: 'Curve left', 20: 'Curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery', 24: 'Narrows right', 25: 'Road work', 
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children', 29: 'Bicycles', 
    30: 'Ice/Snow', 31: 'Wild animals', 32: 'End limits', 33: 'Turn right', 
    34: 'Turn left', 35: 'Ahead only', 36: 'Straight/Right', 37: 'Straight/Left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout', 41: 'End no passing', 
    42: 'End no pass >3.5t'
}

def load_gtsrb_data(base_path, mode="train", sample_fraction=1.0):
    """Loads GTSRB images and labels from disk.

    Args:
        base_path: Path to the dataset root folder.
        mode: The loading mode. "train" expects class subfolders; "test" expects flat images and a CSV.
        sample_fraction: Fraction of data to load (0.0 to 1.0) for faster debugging.

    Returns:
        A tuple containing (images, labels) as numpy arrays.
    """
    print(f"Loading GTSRB ({mode}) from {base_path}...")
    images, labels = [], []
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Path not found: {base_path}")

    # LOGIC FOR TRAINING DATA (based on subfolders)
    if mode == "train":
        classes = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        for class_id, class_name in enumerate(classes):
            class_dir = os.path.join(base_path, class_name)
            img_files = [f for f in os.listdir(class_dir) if f.endswith(('.ppm', '.png'))]
            
            # Apply Sampling
            if sample_fraction < 1.0:
                count = int(len(img_files) * sample_fraction)
                img_files = random.sample(img_files, max(1, count))

            for img_file in img_files:
                img = cv2.imread(os.path.join(class_dir, img_file))
                if img is not None:
                    images.append(cv2.resize(img, IMG_SIZE))
                    labels.append(class_id)

    # LOGIC FOR TEST DATA (Flat folder + CSV Answer Key)
    elif mode == "test":
        # Search for CSV
        csv_file = None
        search_paths = [base_path, os.path.dirname(base_path)]
        for p in search_paths:
            for f in os.listdir(p):
                if f.endswith(".csv") and "test" in f.lower():
                    csv_file = os.path.join(p, f)
                    break
            if csv_file: break
        
        if csv_file is None:
            raise FileNotFoundError(f"No CSV file found in Test folder. Please copy 'GT-final_test.csv' into: {base_path}")
                
        df = pd.read_csv(csv_file, sep=';')

        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42)

        for index, row in df.iterrows():
            img_name = row['Filename']
            class_id = row['ClassId']
            img_path = os.path.join(base_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(cv2.resize(img, IMG_SIZE))
                labels.append(class_id)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'train' or 'test'.")

    return np.array(images), np.array(labels)

def compute_mean_std_from_loader(base_path, split="train", sample_fraction=1.0):
    """
    Computes per-channel mean and std for GTSRB using the existing loader above: load_gtsrb_data().
    """
    images, _ = load_gtsrb_data(
        base_path=base_path,
        mode=split,
        sample_fraction=sample_fraction
    )

    # Accumulators
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    num_pixels = 0

    for img in images:
        # img is H x W x 3 (AND in BGR from cv2 still!)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to float tensor in [0,1]
        img = torch.tensor(img, dtype=torch.float32) / 255.0 # uint8 from cv2 reading

        # Reorder to Channels x H x W
        img = img.permute(2, 0, 1)

        pixels = img.numel() / 3
        num_pixels += pixels

        channel_sum += img.sum(dim=[1, 2])
        channel_sq_sum += (img ** 2).sum(dim=[1, 2])

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)

    return mean, std


class GTSRB(Dataset):
    '''
    To make the dataset loader compatible with PyTorch's DataLoader class, we create a custom Dataset class.
    This class loads the GTSRB data with the above defined methods and implements the required methods 
    __len__ and __getitem__.
    '''
    def __init__(self, base_path, split: str = "train", sample_fraction=1.0, transform=None):
        self.images, self.labels = load_gtsrb_data(base_path=base_path, mode=split, sample_fraction=sample_fraction)
        self.transform = transform

        # source: self extracted from training set with helper function above
        self.mean = torch.tensor([0.3779, 0.3472, 0.3561]).view(3,1,1)
        self.std  = torch.tensor([0.3005, 0.2944, 0.3008]).view(3,1,1)

        # to get a tensor and as ToTensor is deprecated: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.ToTensor.html
        self.base_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx] # H x W x 3 (uint8) 
        label = self.labels[idx]
        
        # OpenCV loads BGR: convert to RGB!
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # convert to tensor
        img = self.base_transform(img)

        if self.transform is not None:
            img = self.transform(img)
        
        img = (img - self.mean) / self.std # now center to properly normalize

        return img, torch.tensor(label, dtype=torch.long)
