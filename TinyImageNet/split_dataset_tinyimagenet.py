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
    """Custom TinyImageNet dataset that handles the validation set properly"""
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        
        if download and not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
            download_tiny_imagenet(root)
        
        self.data = []
        self.targets = []
        
        # Create class to index mapping
        self.class_to_idx = {}
        self.classes = []
        
        if train:
            self._load_train_data()
        else:
            self._load_val_data()
    
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
    
    def _load_val_data(self):
        """Load validation data"""
        val_dir = os.path.join(self.root, "tiny-imagenet-200", "val")
        val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
        
        # First, load class names from train directory to maintain consistency
        train_dir = os.path.join(self.root, "tiny-imagenet-200", "train")
        class_dirs = sorted([d for d in os.listdir(train_dir) 
                           if os.path.isdir(os.path.join(train_dir, d))])
        
        for idx, class_name in enumerate(class_dirs):
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
        
        # Load validation annotations
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                
                img_path = os.path.join(val_dir, "images", img_name)
                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.targets.append(self.class_to_idx[class_name])
    
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

def split_client_task_tinyimagenet(dataset, y_list, client_num, task_num, class_each_task):
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
        # For TinyImageNet, we want the same 180 classes for all clients
        # but in different task orders
        client_classes = y_set[:task_num * class_each_task].copy()  # Use first 180 classes
        random.shuffle(client_classes)
        
        # Select classes for each task (no overlap within client)
        client_i_tasks = []
        for task_i in range(task_num):
            start_idx = task_i * class_each_task
            end_idx = start_idx + class_each_task
            task_classes = client_classes[start_idx:end_idx]
            client_i_tasks.append(task_classes)
        
        client_y_list.append(np.array(client_i_tasks))
    
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
                print(f"Client {c_i}, Task {t_i}, Class {y_c_t}: {len(y_ind_c_t)} samples")
                
                # Shuffle indices
                y_ind_c_t_shuffled = y_ind_c_t.copy()
                np.random.shuffle(y_ind_c_t_shuffled)
                
                # For TinyImageNet, use 100 samples per class per client for train
                # This gives 3000 samples per task (30 classes * 100 samples)
                each_client_data_num = min(100, len(y_ind_c_t_shuffled))
                client_t_ind.extend(y_ind_c_t_shuffled[:each_client_data_num].tolist())
            
            client_ind.append(client_t_ind)
            client_ind_len.append(len(client_t_ind))
            print(f'Client {c_i}, Task {t_i}: {len(client_t_ind)} total samples')
        
        client_ind_list.append(client_ind)
        client_ind_list_len.append(client_ind_len)
    
    client_ind_list_len = np.array(client_ind_list_len)
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
    
    # Load train and test datasets using custom dataset class
    data_train = TinyImageNetDataset(args.datadir, train=True, transform=transform, download=True)
    data_test = TinyImageNetDataset(args.datadir, train=False, transform=transform, download=True)
    
    print(f"Train samples: {len(data_train)}, Test samples: {len(data_test)}")
    
    # Get labels
    train_y_list = data_train.targets
    test_y_list = data_test.targets
    
    # Split data
    print("\nSplitting data for federated continual learning...")
    train_inds, client_y_list = split_client_task_tinyimagenet(
        args.dataset, train_y_list, args.client_num, args.task_num, args.class_each_task
    )
    
    # For test data, use a smaller number of samples per class
    test_inds, _ = split_client_task_tinyimagenet(
        args.dataset, test_y_list, args.client_num, args.task_num, args.class_each_task
    )
    
    # Save split information
    pickle_dict = {
        'train_inds': train_inds, 
        'test_inds': test_inds, 
        'client_y_list': client_y_list
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
    
    # Verify the split
    print("\nVerifying data split...")
    for c_i in range(min(3, args.client_num)):  # Check first 3 clients
        print(f"\nClient {c_i}:")
        for t_i in range(args.task_num):
            classes = client_y_list[c_i][t_i]
            print(f"  Task {t_i}: Classes {classes[:5]}... (showing first 5 of {len(classes)})")

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