#!/usr/bin/env python
import argparse
import os
import torchvision.datasets as datasets
from torchvision import transforms
import random
import numpy as np
import pickle
from utils.utils import setup_seed
import urllib.request
import zipfile
import shutil
from PIL import Image
import torch
import torch.utils.data as data

def download_tiny_imagenet(data_dir):
    """Download and extract Tiny ImageNet dataset"""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file = os.path.join(data_dir, "tiny-imagenet-200.zip")
    
    if not os.path.exists(os.path.join(data_dir, "tiny-imagenet-200")):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_file)
        
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up zip file
        os.remove(zip_file)
        print("Tiny ImageNet downloaded and extracted successfully!")
    else:
        print("Tiny ImageNet already exists.")

class TinyImageNetDataset(data.Dataset):
    """Custom TinyImageNet dataset that uses train split for both train and test"""
    def __init__(self, root, train=True, transform=None, download=False, indices=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.indices = indices
        
        if download and not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
            download_tiny_imagenet(root)
        
        self.data = []
        self.targets = []
        
        # Create class to index mapping
        self.class_to_idx = {}
        self.classes = []
        
        # Always load from training data
        self._load_train_data()
        
        # If indices are provided, use only those indices
        if self.indices is not None:
            self.data = [self.data[i] for i in self.indices]
            self.targets = [self.targets[i] for i in self.indices]
    
    def _load_train_data(self):
        """Load training data"""
        train_dir = os.path.join(self.root, "tiny-imagenet-200", "train")
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
        
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            
            class_dir = os.path.join(train_dir, class_name, "images")
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.JPEG', '.jpeg', '.jpg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append(img_path)
                        self.targets.append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

def split_client_task_tinyimagenet(dataset, y_list, client_num, task_num, class_each_task, is_test=False):
    """
    Split TinyImageNet dataset for federated continual learning
    - 10 clients
    - 6 tasks per client
    - 30 classes per task
    - No class overlap between tasks within each client
    """
    # TinyImageNet has 200 classes (0-199)
    total_classes = 200
    y_set = list(range(total_classes))
    
    assert task_num * class_each_task <= total_classes, f"Need {task_num * class_each_task} classes but only have {total_classes}"
    
    # Create client-task-class assignment
    client_y_list = []
    
    for client_i in range(client_num):
        # Use first 180 classes for all clients but in different task orders
        client_classes = y_set[:task_num * class_each_task].copy()
        random.shuffle(client_classes)
        
        # Select classes for each task (no overlap within client)
        client_i_tasks = []
        for task_i in range(task_num):
            start_idx = task_i * class_each_task
            end_idx = start_idx + class_each_task
            task_classes = client_classes[start_idx:end_idx]
            client_i_tasks.append(task_classes)
        
        client_y_list.append(client_i_tasks)
    
    # Create index lists for train and test data
    y_ind_dict = {}
    for y in y_set:
        y_ind_dict[y] = np.where(np.array(y_list) == y)[0]
    
    y_list = np.array(y_list)
    client_ind_list = []
    client_ind_list_len = []
    
    for c_i in range(client_num):
        client_ind = []
        client_ind_len = []
        
        for t_i in range(task_num):
            client_t_ind = []
            
            for y_c_t in client_y_list[c_i][t_i]:
                y_ind_c_t = y_ind_dict[y_c_t]
                
                # Shuffle indices
                y_ind_c_t_shuffled = y_ind_c_t.copy()
                np.random.shuffle(y_ind_c_t_shuffled)
                
                # Different sample sizes for train and test
                if is_test:
                    # For test, use 20% of available samples (max 20 per class)
                    start_idx = int(0.8 * len(y_ind_c_t_shuffled))
                    each_client_data_num = min(20, len(y_ind_c_t_shuffled) - start_idx)
                    selected_indices = [int(idx) for idx in y_ind_c_t_shuffled[start_idx:start_idx + each_client_data_num]]
                else:
                    # For train, use 80% of available samples (max 80 per class)
                    each_client_data_num = min(80, int(0.8 * len(y_ind_c_t_shuffled)))
                    selected_indices = [int(idx) for idx in y_ind_c_t_shuffled[:each_client_data_num]]
                
                client_t_ind.extend(selected_indices)
            
            client_ind.append(client_t_ind)
            client_ind_len.append(len(client_t_ind))
            
            if c_i == 0:  # Print info for first client only
                print(f'Client {c_i}, Task {t_i}: {len(client_t_ind)} {"test" if is_test else "train"} samples')
        
        client_ind_list.append(client_ind)
        client_ind_list_len.append(client_ind_len)
    
    return client_ind_list, client_y_list

def main(args):
    # Create data directory if it doesn't exist
    os.makedirs(args.datadir, exist_ok=True)
    
    # Load TinyImageNet dataset
    print("Loading TinyImageNet dataset...")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(64),  # TinyImageNet images are 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    full_dataset = TinyImageNetDataset(args.datadir, train=True, transform=transform, download=True)
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Get all labels
    all_y_list = full_dataset.targets
    
    # Create a train/test split from the full dataset
    # Use the same indices for both train and test splitting to ensure consistency
    print("\nCreating train/test split from training data...")
    
    # Split data for training (using first 80% of data)
    train_inds, client_y_list = split_client_task_tinyimagenet(
        args.dataset, all_y_list, args.client_num, args.task_num, args.class_each_task, is_test=False
    )
    
    # Split data for testing (using last 20% of data)
    test_inds, _ = split_client_task_tinyimagenet(
        args.dataset, all_y_list, args.client_num, args.task_num, args.class_each_task, is_test=True
    )
    
    # Verify and convert indices
    def verify_and_convert_indices(indices_list):
        """Ensure all indices are Python integers"""
        converted = []
        for client_indices in indices_list:
            client_converted = []
            for task_indices in client_indices:
                task_converted = [int(idx) for idx in task_indices]
                client_converted.append(task_converted)
            converted.append(client_converted)
        return converted
    
    train_inds = verify_and_convert_indices(train_inds)
    test_inds = verify_and_convert_indices(test_inds)
    
    # Convert client_y_list to ensure all are Python integers
    client_y_list_clean = []
    for client in client_y_list:
        client_tasks = []
        for task in client:
            task_classes = [int(c) for c in task]
            client_tasks.append(task_classes)
        client_y_list_clean.append(client_tasks)
    
    # Save split information
    pickle_dict = {
        'train_inds': train_inds, 
        'test_inds': test_inds, 
        'client_y_list': client_y_list_clean
    }
    
    # Create data_split directory if it doesn't exist
    os.makedirs(os.path.dirname(args.data_split_file), exist_ok=True)
    
    with open(args.data_split_file, "wb") as f:
        pickle.dump(pickle_dict, f)
    
    print(f"\nData split saved to {args.data_split_file}")
    
    # Print summary
    print("\nData Split Summary:")
    print(f"- Number of clients: {args.client_num}")
    print(f"- Tasks per client: {args.task_num}")
    print(f"- Classes per task: {args.class_each_task}")
    print(f"- Total classes used: {args.task_num * args.class_each_task} out of 200")
    
    # Calculate total samples
    total_train_samples = sum(len(indices) for client in train_inds for indices in client)
    total_test_samples = sum(len(indices) for client in test_inds for indices in client)
    print(f"- Total train samples: {total_train_samples}")
    print(f"- Total test samples: {total_test_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TinyImageNet")
    parser.add_argument("--datadir", type=str, default="./datasets/PreciseFCL/")
    parser.add_argument("--data_split_file", type=str, default="data_split/TinyImageNet_split_cn10_tn6_cet30_s42.pkl")
    parser.add_argument("--client_num", type=int, default=10)
    parser.add_argument("--task_num", type=int, default=6)
    parser.add_argument("--class_each_task", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    setup_seed(args.seed)
    main(args)