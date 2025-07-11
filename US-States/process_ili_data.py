# preprocess_ili_data.py
import numpy as np
import os
import pickle
import torch

def preprocess_ili_data(input_path='data/state360.txt', output_dir='data/processed'):
    """
    Preprocess the ILI data for federated learning with AF-FCL
    
    Structure:
    - 7 clients, each with 7 states
    - 6 tasks per client
    - 50 rows per task for training
    - 60 rows for testing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print(f"Loading data from {input_path}")
    data = []
    with open(input_path, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            data.append(values)
    
    data = np.array(data, dtype=np.float32)
    print(f"Loaded data shape: {data.shape}")  # Should be (360, 49)
    
    # Normalize data
    data_min = data.min()
    data_max = data.max()
    data_normalized = (data - data_min) / (data_max - data_min + 1e-8)
    
    # Split into train and test
    train_data_full = data_normalized[:300]  # First 300 rows
    test_data_full = data_normalized[300:]   # Last 60 rows
    
    print(f"Train data shape: {train_data_full.shape}")  # (300, 49)
    print(f"Test data shape: {test_data_full.shape}")    # (60, 49)
    
    # Assign states to clients (7 states per client)
    client_assignments = {}
    for client_id in range(7):
        start_state = client_id * 7
        end_state = start_state + 7
        client_assignments[client_id] = list(range(start_state, end_state))
        print(f"Client {client_id}: states {client_assignments[client_id]}")
    
    # Create AF-FCL compatible data structure
    train_data_dict = {}
    test_data_dict = {}
    
    for client_id in range(7):
        client_states = client_assignments[client_id]
        client_name = f'client_{client_id}'
        
        # Extract client's data (only their states)
        client_train = train_data_full[:, client_states]  # (300, 7)
        client_test = test_data_full[:, client_states]    # (60, 7)
        
        # Split training data into tasks
        x_tasks = []
        y_tasks = []
        
        for task_id in range(6):
            start_row = task_id * 50
            end_row = start_row + 50
            
            task_data = client_train[start_row:end_row]  # (50, 7)
            
            # Create samples for this task
            task_x = []
            task_y = []
            
            # For time series, we can use sliding windows or direct values
            # Here we'll use direct values with state classification
            for i, row in enumerate(task_data):
                # Option 1: Use the full state vector as input
                task_x.append(row)
                
                # Option 2: Create labels based on dominant state or pattern
                # For classification: use the state with highest value
                dominant_state = client_states[np.argmax(row)]
                task_y.append(dominant_state)
            
            x_tasks.append(task_x)
            y_tasks.append(task_y)
        
        # Process test data
        # For continual learning, we typically use the same test set for all tasks
        test_x_all = []
        test_y_all = []
        
        for i, row in enumerate(client_test):
            test_x_all.append(row)
            dominant_state = client_states[np.argmax(row)]
            test_y_all.append(dominant_state)
        
        # Replicate test data for each task (AF-FCL expects this format)
        test_x_tasks = [test_x_all for _ in range(6)]
        test_y_tasks = [test_y_all for _ in range(6)]
        
        # Store in AF-FCL format
        train_data_dict[client_name] = {
            'x': x_tasks,  # List of 6 tasks, each with 50 samples
            'y': y_tasks   # List of 6 tasks, each with 50 labels
        }
        
        test_data_dict[client_name] = {
            'x': test_x_tasks,  # List of 6 tasks, each with test samples
            'y': test_y_tasks   # List of 6 tasks, each with test labels
        }
    
    # Create the final data structure expected by AF-FCL
    ili_data = {
        'client_names': [f'client_{i}' for i in range(7)],
        'train_data': train_data_dict,
        'test_data': test_data_dict,
        'unique_labels': 49,  # Total number of states
        'normalization': {'min': data_min, 'max': data_max},
        'client_assignments': client_assignments,
        'input_dim': 7,  # Each client sees 7 states
        'num_tasks': 6,
        'samples_per_task': 50
    }
    
    # Also create simple pickle format
    ili_processed = {
        'train_data': train_data_full,
        'test_data': test_data_full,
        'client_assignments': client_assignments,
        'task_splits': {i: {'start': i*50, 'end': (i+1)*50} for i in range(6)}
    }
    
    # Save in pickle format
    output_path = os.path.join(output_dir, 'ili_afcl_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ili_data, f)
    
    # Save processed data too
    processed_path = os.path.join(output_dir, 'ili_processed.pkl')
    with open(processed_path, 'wb') as f:
        pickle.dump(ili_processed, f)
    
    print(f"\nProcessed data saved to {output_path}")
    print(f"Simple processed data saved to {processed_path}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Number of clients: 7")
    print(f"States per client: 7")
    print(f"Number of tasks: 6")
    print(f"Training samples per task: 50")
    print(f"Test samples: 60")
    print(f"Total unique labels: 49")
    
    # Verify data structure
    print("\nVerifying data structure:")
    for client_name in ili_data['client_names']:
        train_client = train_data_dict[client_name]
        test_client = test_data_dict[client_name]
        print(f"\n{client_name}:")
        print(f"  Training tasks: {len(train_client['x'])}")
        for t_idx in range(len(train_client['x'])):
            print(f"    Task {t_idx}: {len(train_client['x'][t_idx])} samples, "
                  f"{len(train_client['y'][t_idx])} labels")
        print(f"  Test tasks: {len(test_client['x'])}")
        print(f"    Test samples per task: {len(test_client['x'][0])}")
    
    return ili_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/state360.txt',
                        help='Path to the state360.txt file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    
    args = parser.parse_args()
    
    preprocess_ili_data(args.input_path, args.output_dir)