import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, img_dir, norm, label2idx, split='train', train_ratio=0.8, img_height=224, img_width=224, random_state=42):
        self.resize = Resize((img_height, img_width))
        self.norm = norm
        self.split = split
        self.train_ratio = train_ratio
        self.img_dir = img_dir
        self.label2idx = label2idx
        self.random_state = random_state

        # Read image file paths and labels
        self.img_paths, self.img_labels = self.read_img_files()

        # Split the data into train and validation sets
        if split in ['train', 'val'] and 'train' in img_dir.lower():
            train_data, val_data = train_test_split(
                list(zip(self.img_paths, self.img_labels)),
                train_size=self.train_ratio,
                random_state=self.random_state,
                stratify=self.img_labels
            )

            if split == 'train':
                self.img_paths, self.img_labels = zip(*train_data)
            elif split == 'val':
                self.img_paths, self.img_labels = zip(*val_data)

    def read_img_files(self):
        """Read image file paths and corresponding labels."""
        img_paths = []
        img_labels = []

        for cls in self.label2idx.keys():
            cls_path = os.path.join(self.img_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for img in os.listdir(cls_path):
                img_paths.append(os.path.join(cls_path, img))
                img_labels.append(cls)

        return img_paths, img_labels

    def __len__(self):
        """Return the total number of samples."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Fetch an image and its corresponding label."""
        img_path = self.img_paths[idx]
        cls = self.img_labels[idx]

        # Read and preprocess the image
        img = self.resize(read_image(img_path))
        img = img.type(torch.float32)

        # Normalize the image if required
        if self.norm:
            img = (img / 127.5) - 1

        # Convert class label to index
        label = self.label2idx[cls]
        return img, label
