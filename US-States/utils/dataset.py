import os
import pickle
from typing import Any
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms
import random
import torch.utils.data as data

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
    if dataset_name == 'ILI':
        from utils.ili_dataset import get_ili_dataset
        return get_ili_dataset(args, datadir, data_split_file)

    
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