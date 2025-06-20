# utils/ili_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os

class ILIDataset(Dataset):
    """Dataset class for ILI time-series data"""
    def __init__(self, data, labels, window_size=5):
        """
        Args:
            data: numpy array of shape (num_samples, num_features)
            labels: numpy array of shape (num_samples,)
            window_size: size of sliding window for time-series
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        # Create sliding window
        x = self.data[idx:idx+self.window_size].flatten()
        # Use the last value as label (next-step prediction)
        y = self.labels[idx+self.window_size-1] if idx+self.window_size-1 < len(self.labels) else self.labels[-1]
        return x, y

def load_ili_data(data_path='data/processed/ili_processed.pkl'):
    """Load preprocessed ILI data"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_ili_dataset(args, datadir, data_split_file):
    """Get ILI dataset in the format expected by AF-FCL"""
    
    # Load preprocessed data
    data = load_ili_data(os.path.join(datadir, 'ili_processed.pkl'))
    
    train_data = data['train_data']  # (300, 49)
    test_data = data['test_data']     # (60, 49)
    client_assignments = data['client_assignments']
    task_splits = data['task_splits']
    
    # Number of unique states (classes)
    unique_labels = 49
    
    # Prepare data structure
    client_names = [f'client_{i}' for i in range(7)]
    train_data_dict = {}
    test_data_dict = {}
    
    for client_id in range(7):
        client_states = client_assignments[client_id]
        
        # Extract client's data (only their states)
        client_train = train_data[:, client_states]  # (300, 7)
        client_test = test_data[:, client_states]    # (60, 7)
        
        # Split into tasks
        x_tasks = []
        y_tasks = []
        
        for task_id in range(6):
            task_info = task_splits[task_id]
            task_data = client_train[task_info['start']:task_info['end']]  # (50, 7)
            
            # Create dataset for this task
            # For simplicity, we'll use the state index as the label
            # In practice, you might want to discretize the values or use regression
            task_x = []
            task_y = []
            
            for row in task_data:
                task_x.append(row)
                # For classification, we'll use the argmax of the row as label
                # This maps each sample to its highest-value state
                task_y.append(client_states[np.argmax(row)])
            
            x_tasks.append(task_x)
            y_tasks.append(task_y)
        
        # Test data (all tasks see the same test data)
        test_x = []
        test_y = []
        for row in client_test:
            test_x.append(row)
            test_y.append(client_states[np.argmax(row)])
        
        # Store in format expected by AF-FCL
        train_data_dict[f'client_{client_id}'] = {
            'x': x_tasks,  # List of 6 tasks, each with 50 samples
            'y': y_tasks   # List of 6 tasks, each with 50 labels
        }
        
        test_data_dict[f'client_{client_id}'] = {
            'x': [test_x] * 6,  # Repeat for each task
            'y': [test_y] * 6   # Repeat for each task
        }
    
    return {
        'client_names': client_names,
        'train_data': train_data_dict,
        'test_data': test_data_dict,
        'unique_labels': unique_labels
    }