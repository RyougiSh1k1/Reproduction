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

class TinyImageNet(datasets.ImageFolder):
    """Custom TinyImageNet dataset class"""
    def __init__(self, root, train=True, transform=None, download=False):
        if download and not os.path.exists(os.path.join(root, "tiny-imagenet-200")):
            download_tiny_imagenet(root)
            
        if train:
            super().__init__(os.path.join(root, "tiny-imagenet-200", "train"), transform=transform)
        else:
            # For test set, we need to reorganize the val folder
            val_dir = os.path.join(root, "tiny-imagenet-200", "val")
            self._prepare_val_folder(val_dir)
            super().__init__(val_dir, transform=transform)
    
    def _prepare_val_folder(self, val_dir):
        """Reorganize validation folder to have class subfolders"""
        val_annotations = os.path.join(val_dir, "val_annotations.txt")
        if not os.path.exists(val_annotations):
            return
            
        # Read annotations
        val_img_dict = {}
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                val_img_dict[parts[0]] = parts[1]
        
        # Create class folders if they don't exist
        for img_name, class_name in val_img_dict.items():
            class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            
            # Move images to class folders
            src = os.path.join(val_dir, "images", img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)

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
        # Randomly shuffle all classes for this client
        client_classes = y_set.copy()
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
                
                # For TinyImageNet, use 100 samples per class per client
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
    
    # Load train and test datasets
    data_train = TinyImageNet(args.datadir, train=True, transform=transform, download=True)
    data_test = TinyImageNet(args.datadir, train=False, transform=transform, download=True)
    
    print(f"Train samples: {len(data_train)}, Test samples: {len(data_test)}")
    
    # Get labels
    train_y_list = [data_train[i][1] for i in range(len(data_train))]
    test_y_list = [data_test[i][1] for i in range(len(data_test))]
    
    # Split data
    print("\nSplitting data for federated continual learning...")
    train_inds, client_y_list = split_client_task_tinyimagenet(
        args.dataset, train_y_list, args.client_num, args.task_num, args.class_each_task
    )
    
    test_inds, _ = split_client_task_tinyimagenet(
        args.dataset, test_y_list, args.client_num, args.task_num, args.class_each_task
    )
    
    # Save split information
    pickle_dict = {
        'train_inds': train_inds, 
        'test_inds': test_inds, 
        'client_y_list': client_y_list.tolist()
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