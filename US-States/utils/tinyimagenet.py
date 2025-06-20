# utils/tinyimagenet.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class TinyImageNetDataset(Dataset):
    """
    TinyImageNet Dataset
    
    Assumes TinyImageNet-200 is downloaded and has the following structure:
    tiny-imagenet-200/
        train/
            n01443537/
                images/
                    n01443537_0.JPEG
                    ...
            ...
        val/
            images/
                val_0.JPEG
                ...
            val_annotations.txt
        wnids.txt
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load class names
        self.class_names = {}
        wnids_path = os.path.join(root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise FileNotFoundError(f"wnids.txt not found at {wnids_path}")
            
        with open(wnids_path, 'r') as f:
            for i, line in enumerate(f):
                self.class_names[line.strip()] = i
        
        self.samples = []
        
        if split == 'train':
            train_dir = os.path.join(root, 'train')
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Train directory not found at {train_dir}")
                
            for class_name, class_idx in self.class_names.items():
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_path in glob.glob(os.path.join(class_dir, '*.JPEG')):
                        self.samples.append((img_path, class_idx))
                        
        elif split == 'val' or split == 'test':
            val_dir = os.path.join(root, 'val')
            if not os.path.exists(val_dir):
                raise FileNotFoundError(f"Val directory not found at {val_dir}")
                
            val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
            if not os.path.exists(val_annotations_file):
                raise FileNotFoundError(f"val_annotations.txt not found at {val_annotations_file}")
                
            with open(val_annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_name = parts[1]
                        img_path = os.path.join(val_dir, 'images', img_name)
                        if class_name in self.class_names and os.path.exists(img_path):
                            self.samples.append((img_path, self.class_names[class_name]))
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split {split}")
            
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (64, 64), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

    def get_class_to_idx(self):
        """Return the mapping of class names to indices"""
        return self.class_names
    
    def get_num_classes(self):
        """Return the number of classes"""
        return len(self.class_names)