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

def testify_client_y_list(y_list, inds, client_y_list):
    y_list = np.array(y_list)
    for c_i in range(len(inds)):
        for t_i in range(len(inds[c_i])):
            y_c_t = y_list[np.array(inds[c_i][t_i])]
            y_c_t_set = set(y_c_t)
            assert y_c_t_set==set(client_y_list[c_i][t_i])

def split_data_from_inds(data, inds):
    data_reshape = {}
    for c_i in range(len(inds)):
        x_c = []
        y_c = []
        for t_i in range(len(inds[c_i])):
            inds_c_t = inds[c_i][t_i]
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
        
        # Load TinyImageNet
        data_train = TinyImageNet(datadir, train=True, transform=transform, download=True)
        data_test = TinyImageNet(datadir, train=False, transform=transform, download=True)

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