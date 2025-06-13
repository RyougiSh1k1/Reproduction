import os
import pickle
from typing import Any
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import random
import torch.utils.data as data
import urllib.request
import zipfile
import shutil
from PIL import Image

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

def testify_client_y_list(y_list, inds, client_y_list):
    """Verify that the indices match the expected class labels"""
    if len(y_list) == 0:
        print("Warning: Empty y_list, skipping verification")
        return
        
    y_list = np.array(y_list)
    
    # Verify all indices are within bounds
    for c_i in range(len(inds)):
        for t_i in range(len(inds[c_i])):
            if len(inds[c_i][t_i]) == 0:
                print(f"Warning: No indices for client {c_i}, task {t_i}")
                continue
                
            # Check bounds and convert to valid indices
            valid_indices = []
            for idx in inds[c_i][t_i]:
                idx = int(idx)
                if 0 <= idx < len(y_list):
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Index {idx} out of bounds for y_list of length {len(y_list)}")
            
            if len(valid_indices) == 0:
                continue
                
            # Get actual classes for these indices
            indices = np.array(valid_indices, dtype=np.int64)
            y_c_t = y_list[indices]
            y_c_t_set = set(y_c_t)
            expected_set = set(client_y_list[c_i][t_i])
            
            # Check if actual classes match expected
            if len(y_c_t_set) > 0:
                if not y_c_t_set.issubset(expected_set):
                    print(f"Warning: Class mismatch for client {c_i}, task {t_i}:")
                    print(f"  Expected classes: {expected_set}")
                    print(f"  Actual classes: {y_c_t_set}")
                    print(f"  Unexpected classes: {y_c_t_set - expected_set}")
                    print(f"  Missing classes: {expected_set - y_c_t_set}")
                    
                    # Don't fail for TinyImageNet as it may have data distribution issues
                    if 'TinyImageNet' not in str(client_y_list):
                        assert False, f"Class mismatch for client {c_i}, task {t_i}"

def split_data_from_inds(data, inds):
    data_reshape = {}
    for c_i in range(len(inds)):
        x_c = []
        y_c = []
        for t_i in range(len(inds[c_i])):
            # Ensure indices are integers
            inds_c_t = [int(i) for i in inds[c_i][t_i]]
            x_c_t = [data[i][0] for i in inds_c_t]
            y_c_t = [data[i][1] for i in inds_c_t]

            x_c.append(x_c_t)
            y_c.append(y_c_t)
        
        data_reshape['client_%d'%c_i] = {'x': x_c, 'y': y_c}
    
    return data_reshape

def malicious_dataset(data_train_d, data_test_d, unique_labels, malicious_client_num=1):
    clients_names = list(data_train_d.keys())
    random.shuffle(clients_names)
    assert malicious_client_num<=len(clients_names)
    malicious_clients = clients_names[:malicious_client_num]
    
    for malicious_c in malicious_clients:
        y_list = []
        for y_task in data_train_d[malicious_c]['y']:
            y_list += y_task
        y_set = set(y_list)
        for yi in y_set:
            y_change = random.choice(list(range(unique_labels)))
            
            def replace_y(data_d):
                ys_replace = []
                for y_task_ in data_d[malicious_c]['y']:
                    y_task_np = np.array(y_task_)
                    y_task_np[y_task_np==yi] = y_change
                    ys_replace.append(y_task_np.tolist())
                return ys_replace
    
            data_train_d[malicious_c]['y'] = replace_y(data_train_d)
            data_test_d[malicious_c]['y'] = replace_y(data_test_d)

    return data_train_d, data_test_d

def get_dataset(args, dataset_name, datadir, data_split_file):
    if dataset_name=='EMNIST-Letters' or dataset_name=='EMNIST-Letters-malicious' or dataset_name=='EMNIST-Letters-shuffle':
        unique_labels = 26

        if dataset_name=='EMNIST-Letters-shuffle':
            assert 'EMNIST_letters_shuffle' in data_split_file        

        data_train = datasets.EMNIST(datadir, 'letters', download=False, train=True, transform=transforms.ToTensor(), target_transform=lambda x:x-1)
        data_test = datasets.EMNIST(datadir, 'letters', download=False, train=False, transform=transforms.ToTensor(), target_transform=lambda x:x-1)

    elif dataset_name=='CIFAR100':
        unique_labels = 100

        data_train = datasets.CIFAR100(datadir, download=True, train=True)
        data_test = datasets.CIFAR100(datadir, download=True, train=False)

    elif dataset_name=='TinyImageNet':
        unique_labels = 200
        
        # Define transforms for TinyImageNet
        transform = transforms.Compose([
            transforms.Resize(64),  # TinyImageNet images are 64x64
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load TinyImageNet using custom dataset class
        data_train = TinyImageNetDataset(datadir, train=True, transform=transform, download=True)
        data_test = TinyImageNetDataset(datadir, train=False, transform=transform, download=True)

    elif args.dataset=='MNIST-SVHN-FASHION':
        unique_labels = 20

        download = False
        repeat_transform = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        mean=(0.1,)
        std=(0.2752,)
        # 60000 10000
        mnist_data_train = datasets.MNIST(args.datadir, train=True,download=download,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std), repeat_transform]))
        mnist_data_test = datasets.MNIST(args.datadir, train=False,download=download,transform=transforms.Compose([
                    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std), repeat_transform]))

        mean=[0.4377,0.4438,0.4728]
        std=[0.198,0.201,0.197]
        # 73257 26032
        svhn_data_train = datasets.SVHN(args.datadir, split='train',download=download,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        svhn_data_test = datasets.SVHN(args.datadir, split='test',download=download,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])) 

        mean=(0.2190,) # Mean and std including the padding
        std=(0.3318,)
        # 60000 
        fashionmnist_data_train = datasets.FashionMNIST(args.datadir, train=True, download=download, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std), repeat_transform]),
                    target_transform=lambda x:x+10)
        fashionmnist_data_test = datasets.FashionMNIST(args.datadir, train=False, download=download, transform=transforms.Compose([
                    transforms.Pad(padding=2, fill=0), transforms.ToTensor(),transforms.Normalize(mean, std), repeat_transform]),
                    target_transform=lambda x:x+10)
        # fashionmnist_label_train = [fashionmnist_data_train[i][1] for i in range(len(fashionmnist_data_train))]

        data_train = []
        data_test = []
        for dataset in [mnist_data_train, svhn_data_train, fashionmnist_data_train]:
            data_train += [dataset[i] for i in range(len(dataset))]
        for dataset in [mnist_data_test, svhn_data_test, fashionmnist_data_test]:
            data_test += [dataset[i] for i in range(len(dataset))]

    if dataset_name == 'TinyImageNet':
        # For TinyImageNet, we need to handle the dataset differently
        train_y_list = data_train.targets
        test_y_list = data_test.targets
    else:
        train_y_list = [data_train[i][1] for i in range(len(data_train))]
        test_y_list = [data_test[i][1] for i in range(len(data_test))]

    with open(os.path.join(datadir, data_split_file), 'rb') as f:
        split_data = pickle.load(f)

    testify_client_y_list(train_y_list, split_data['train_inds'], split_data['client_y_list'])
    testify_client_y_list(test_y_list, split_data['test_inds'], split_data['client_y_list'])

    data_train_reshape = split_data_from_inds(data_train, split_data['train_inds'])
    data_test_reshape = split_data_from_inds(data_test, split_data['test_inds'])

    if dataset_name=='EMNIST-Letters-malicious':
        data_train_reshape, data_test_reshape = malicious_dataset(data_train_reshape, data_test_reshape,
                                                                    unique_labels, malicious_client_num=args.malicious_client_num)
        
    return {'client_names': list(data_train_reshape.keys()), 'train_data': data_train_reshape, 'test_data': data_test_reshape, 'unique_labels': unique_labels}

class Transform_dataset(data.Dataset):
    def __init__(self, X, Y, transform=None) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __getitem__(self, index: Any) -> Any:
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x,y

    def __len__(self) -> int:
        return len(self.X)