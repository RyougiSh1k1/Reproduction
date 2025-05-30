import glog as logger
import torch
import time
import numpy as np
import os
from torch.utils.data import DataLoader

from FLAlgorithms.users.userPreciseFCL import UserPreciseFCL
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from utils.dataset import get_dataset
from utils.model_utils import read_user_data_PreciseFCL
from utils.utils import str_in_list
from inception_score import calculate_inception_score_for_classes

class FedPrecise(Server):
    def __init__(self, args, model:PreciseModel, seed):
        super().__init__(args, model, seed)

        self.classifier_global_mode = args.classifier_global_mode
        self.use_adam = 'adam' in self.algorithm.lower()
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        self.unique_labels = self.data['unique_labels']
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']
        self.init_users(self.data, args, model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('Using device: ' + str(device))
        self.device = device
        
        
        for u in self.users:
            u.model = u.model.to(device)

        # server model:
        self.model.to(device)
        # self.gaussian_intiailize(self.model.classifier)

        # Initialize inception score tracking
        self.inception_scores = {}
        self.client_inception_scores = {}  # New: For client-specific scores
        self.avg_inception_scores = {}     # New: For average scores across clients

    def init_users(self, data, args, model):
        self.users = []
        total_users = len(data['client_names'])
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_PreciseFCL(i, data, dataset=args.dataset, count_labels=True, task = 0)

            # count total samples (accumulative)
            self.total_train_samples +=len(train_data)
            self.total_test_samples += len(test_data)
            id = i
            
            # ============ initialize Users with data =============
            
            user=UserPreciseFCL(
                args,
                id, 
                model, 
                train_data, 
                test_data, 
                label_info,
                use_adam=self.use_adam,
                my_model_name='fedprecise',
                unique_labels=self.unique_labels,
                classifier_head_list=self.classifier_head_list,
            )
            
            self.users.append(user)
            
            # update classes so far & current labels
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])
        
        logger.info("Number of Train/Test samples: %d/%d"%(self.total_train_samples, self.total_test_samples))
        logger.info("Data from {} users in total.".format(total_users))
        logger.info("Finished creating FedAvg server.")

    def calculate_client_inception_scores(self, task_id, glob_iter):
        """Calculate inception scores for each client for the current task and round"""
        logger.info("Calculating client-specific inception scores for task {} round {}".format(task_id, glob_iter))
        
        client_scores = {}
        client_stds = {}
        client_class_scores = {}
        
        for user_id, user in enumerate(self.users):
            try:
                # Get the test data for the current task
                if task_id < len(user.test_data_so_far_loader):
                    loader = user.test_data_so_far_loader[task_id]
                    
                    # Skip if no test data available
                    if len(loader.dataset) < 10:  # Minimum samples needed
                        logger.warning(f"Insufficient test data for client {user_id} on task {task_id}")
                        continue
                    
                    # Get all unique labels in this client's dataset
                    all_labels = set()
                    for _, y in loader:
                        all_labels.update(y.numpy())
                    
                    # Calculate inception scores for this client
                    try:
                        overall_score, overall_std, class_scores = calculate_inception_score_for_classes(
                            loader, 
                            list(all_labels),  # Use the actual labels present in the dataset
                            device=self.device
                        )
                        
                        client_scores[user_id] = overall_score
                        client_stds[user_id] = overall_std
                        client_class_scores[user_id] = class_scores
                        
                        logger.info(f"Client {user_id}, Task {task_id}, Round {glob_iter}: " 
                                    f"Inception Score = {overall_score:.4f} ± {overall_std:.4f}")
                    except Exception as e:
                        logger.error(f"Error calculating inception score for client {user_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing client {user_id}: {str(e)}")
        
        # Calculate average inception score across all clients
        if client_scores:
            avg_score = sum(client_scores.values()) / len(client_scores)
            avg_std = sum(client_stds.values()) / len(client_stds)
            logger.info(f"Average across all clients for Task {task_id}, Round {glob_iter}: "
                        f"Inception Score = {avg_score:.4f} ± {avg_std:.4f}")
        else:
            avg_score = None
            avg_std = None
            logger.warning(f"No valid client scores available for task {task_id}, round {glob_iter}")
        
        return client_scores, client_stds, client_class_scores, avg_score, avg_std

    def calculate_inception_scores(self, task_id, glob_iter):
        """
        Calculate combined inception scores for the current task and round
        and also client-specific scores with their average
        """
        logger.info("Calculating inception scores for task {} round {}".format(task_id, glob_iter))
        
        try:
            # Get client-specific scores and average
            client_scores, client_stds, client_class_scores, avg_score, avg_std = self.calculate_client_inception_scores(task_id, glob_iter)
            
            # Create a combined test dataset as in the original method
            all_test_data = []
            all_labels = set()
            
            for user in self.users:
                if task_id < len(user.test_data_so_far_loader):
                    loader = user.test_data_so_far_loader[task_id]
                    for x, y in loader:
                        all_test_data.append((x, y))
                        all_labels.update(y.numpy())
            
            if not all_test_data:
                logger.warning("No test data available for task {}".format(task_id))
                return None, None, None, client_scores, client_stds, avg_score, avg_std
            
            # Create a combined dataloader
            combined_x = torch.cat([batch[0] for batch in all_test_data], dim=0)
            combined_y = torch.cat([batch[1] for batch in all_test_data], dim=0)
            
            combined_dataloader = DataLoader(
                [(combined_x[i], combined_y[i]) for i in range(len(combined_x))],
                batch_size=32,
                shuffle=False
            )
            
            # Calculate inception scores on combined data
            try:
                overall_score, overall_std, class_scores = calculate_inception_score_for_classes(
                    combined_dataloader, 
                    list(all_labels),  # Use the actual labels present in the dataset
                    device=self.device
                )
                
                logger.info(f"Combined data for Task {task_id}, Round {glob_iter}: "
                            f"Inception Score = {overall_score:.4f} ± {overall_std:.4f}")
            except Exception as e:
                logger.error(f"Error calculating combined inception score: {str(e)}")
                overall_score, overall_std, class_scores = None, None, None
            
            return overall_score, overall_std, class_scores, client_scores, client_stds, avg_score, avg_std
        
        except Exception as e:
            logger.error(f"Error in calculate_inception_scores: {str(e)}")
            # Return empty results to avoid breaking the training loop
            return None, None, None, {}, {}, None, None

    def train(self, args):
        
        N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        
        for task in range(N_TASKS):
            self.inception_scores[task] = {}
            self.client_inception_scores[task] = {}  # New: Store client scores
            self.avg_inception_scores[task] = {}     # New: Store average scores
        
        for task in range(N_TASKS):
            
            # ===================
            # The First Task
            # ===================
            if task == 0:

                # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                    
                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ===================
            # Initialize new Task
            # ===================
            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.users)):

                    id, train_data, test_data, label_info = read_user_data_PreciseFCL(i, self.data, dataset=args.dataset, count_labels=True, task = task)

                    # update dataset 
                    self.users[i].next_task(train_data, test_data, label_info)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.users[0].available_labels
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ===================
            #    print info.
            # ===================
            if True:
                for u in self.users:
                    logger.info("classes so far: " + str(u.classes_so_far))
                logger.info("available labels for the Client: " + str(self.users[-1].available_labels))
                logger.info("available labels (current) for the Client: " + str(self.users[-1].available_labels_current))
            

            # 1. server side:
            # 2. user side:
            # ============ train ==============

            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_task in range(epoch_per_task):
                
                glob_iter = glob_iter_task + (epoch_per_task) * task

                logger.info("\n\n------------- Round number: %d | Current task: %d -------------\n\n"%(glob_iter, task))

                # select users
                # self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
                self.selected_users, self.user_idxs = self.select_users(glob_iter, len(self.users), return_idx=True)

                # broadcast averaged prediction model to clients
                if self.algorithm != 'local':                    
                    # send parameteres: server -> client
                    self.send_parameters(mode='all', beta=1)

                chosen_verbose_user = np.random.randint(0, len(self.users))
                self.timestamp = time.time() # log user-training start time

                # ---------------
                #   train user
                # ---------------

                self.pickle_record['train'][glob_iter] = {}

                global_classifier = self.model.classifier
                global_classifier.eval()
                
                for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                    # verbose = user_id == chosen_verbose_user
                    verbose = True
                    
                    # perform regularization using generated samples after the first communication round
                    user_result = user.train(
                        glob_iter,
                        glob_iter_task,
                        global_classifier,
                        verbose=verbose)
                        
                    self.pickle_record['train'][glob_iter][user_id] = user_result

                # log training time
                curr_timestamp = time.time()
                train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
                self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time() # log server-agg start time

                #=================
                # 2. Server update
                #=================

                if self.algorithm != 'local':
                    self.aggregate_parameters_(class_partial=False)
                    
                curr_timestamp=time.time()  # log server-agg end time
                agg_time = curr_timestamp - self.timestamp
                self.metrics['server_agg_time'].append(agg_time)
                
                # When calculating and storing inception scores in the training loop:
                overall_score, overall_std, class_scores, client_scores, client_stds, avg_score, avg_std = self.calculate_inception_scores(task, glob_iter)
                
                if overall_score is not None:
                    self.inception_scores[task][glob_iter] = {
                        'overall_score': overall_score,
                        'overall_std': overall_std,
                        'class_scores': class_scores
                    }
                    
                    # Store client-specific scores and average
                    if avg_score is not None:
                        self.client_inception_scores[task][glob_iter] = {
                            'client_scores': client_scores,
                            'client_stds': client_stds
                        }
                        
                        self.avg_inception_scores[task][glob_iter] = {
                            'avg_score': avg_score,
                            'avg_std': avg_std
                        }
                        
                        # Add inception scores to pickle record
                        if 'inception_scores' not in self.pickle_record:
                            self.pickle_record['inception_scores'] = {}
                        if task not in self.pickle_record['inception_scores']:
                            self.pickle_record['inception_scores'][task] = {}
                        
                        self.pickle_record['inception_scores'][task][glob_iter] = {
                            'overall_score': float(overall_score),
                            'overall_std': float(overall_std),
                            'class_scores': {k: (float(v[0]), float(v[1])) for k, v in class_scores.items()},
                            'client_scores': {k: float(v) for k, v in client_scores.items()},
                            'client_stds': {k: float(v) for k, v in client_stds.items()},
                            'avg_score': float(avg_score),
                            'avg_std': float(avg_std)
                        }
                    
                    # Save inception score visualization
                    self.save_inception_score_visualization(task)

            if self.algorithm != 'local':
                # send parameteres: server -> client
                self.send_parameters(mode='all', beta=1)

            self.evaluate_all_(glob_iter=glob_iter, matrix=True, personal=False)

            self.save_pickle()
            
            # Save final inception scores for this task
            #self.save_inception_score_visualization(task)
            # After finishing a task, save the current inception scores to CSV
            self.save_inception_scores_to_csv()
    
    def save_inception_score_visualization(self, task):
        """Save visualization of inception scores for a given task, including client averages"""
        import matplotlib.pyplot as plt
        
        if task not in self.inception_scores or not self.inception_scores[task]:
            return
            
        # Create directory for visualizations
        vis_dir = os.path.join(self.args.target_dir_name, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        rounds = sorted(self.inception_scores[task].keys())
        scores = [self.inception_scores[task][r]['overall_score'] for r in rounds]
        stds = [self.inception_scores[task][r]['overall_std'] for r in rounds]
        
        # Extract average scores
        avg_scores = [self.avg_inception_scores[task][r]['avg_score'] for r in rounds]
        avg_stds = [self.avg_inception_scores[task][r]['avg_std'] for r in rounds]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(rounds, scores, yerr=stds, marker='o', linestyle='-', label='Combined Data')
        plt.errorbar(rounds, avg_scores, yerr=avg_stds, marker='s', linestyle='--', 
                    label='Avg Across Clients', color='red')
        plt.title(f'Inception Score Progress - Task {task}')
        plt.xlabel('Round')
        plt.ylabel('Inception Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'inception_score_task_{task}.png'))
        plt.close()
        
        # Create client comparison plot
        if task in self.client_inception_scores and self.client_inception_scores[task]:
            plt.figure(figsize=(12, 8))
            
            # Get all clients that have data in any round
            all_clients = set()
            for r in rounds:
                if r in self.client_inception_scores[task]:
                    all_clients.update(self.client_inception_scores[task][r]['client_scores'].keys())
            all_clients = sorted(list(all_clients))
            
            # Plot each client's scores
            for client_id in all_clients:
                client_rounds = []
                client_scores = []
                
                for r in rounds:
                    if r in self.client_inception_scores[task] and client_id in self.client_inception_scores[task][r]['client_scores']:
                        client_rounds.append(r)
                        client_scores.append(self.client_inception_scores[task][r]['client_scores'][client_id])
                
                if client_scores:
                    plt.plot(client_rounds, client_scores, marker='.', linestyle='-', alpha=0.5, 
                            label=f'Client {client_id}')
            
            # Plot the average for comparison
            plt.plot(rounds, avg_scores, marker='s', linestyle='-', linewidth=2, 
                    label='Avg Across Clients', color='red')
            
            plt.title(f'Client-Specific Inception Scores - Task {task}')
            plt.xlabel('Round')
            plt.ylabel('Inception Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'client_inception_scores_task_{task}.png'))
            plt.close()

        """
    Function to save inception scores to CSV files.
    This can be added to the FedPrecise class in serverPreciseFCL.py
    """

    def save_inception_scores_to_csv(self):
        """
        Save inception scores to CSV files for easier analysis.
        Creates separate CSV files for:
        1. Overall inception scores (combined data)
        2. Average inception scores across clients
        3. Client-specific inception scores
        """
        import csv
        import os
        
        # Create directory for CSV files
        csv_dir = os.path.join(self.args.target_dir_name, 'csv_results')
        os.makedirs(csv_dir, exist_ok=True)
        
        # Get all tasks
        tasks = sorted(self.inception_scores.keys())
        
        # 1. Save overall inception scores (combined data)
        with open(os.path.join(csv_dir, 'overall_inception_scores.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Task', 'Round', 'Overall_Score', 'Std_Dev'])
            
            # Write data
            for task in tasks:
                rounds = sorted(self.inception_scores[task].keys())
                for r in rounds:
                    writer.writerow([
                        task, 
                        r, 
                        self.inception_scores[task][r]['overall_score'],
                        self.inception_scores[task][r]['overall_std']
                    ])
        
        # 2. Save average inception scores across clients
        with open(os.path.join(csv_dir, 'avg_inception_scores.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Task', 'Round', 'Avg_Score', 'Avg_Std_Dev'])
            
            # Write data
            for task in tasks:
                if task not in self.avg_inception_scores:
                    continue
                    
                rounds = sorted(self.avg_inception_scores[task].keys())
                for r in rounds:
                    writer.writerow([
                        task, 
                        r, 
                        self.avg_inception_scores[task][r]['avg_score'],
                        self.avg_inception_scores[task][r]['avg_std']
                    ])
        
        # 3. Save client-specific inception scores
        with open(os.path.join(csv_dir, 'client_inception_scores.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Task', 'Round', 'Client_ID', 'Score', 'Std_Dev'])
            
            # Write data
            for task in tasks:
                if task not in self.client_inception_scores:
                    continue
                    
                rounds = sorted(self.client_inception_scores[task].keys())
                for r in rounds:
                    client_scores = self.client_inception_scores[task][r]['client_scores']
                    client_stds = self.client_inception_scores[task][r]['client_stds']
                    
                    for client_id, score in client_scores.items():
                        std = client_stds.get(client_id, 0)
                        writer.writerow([task, r, client_id, score, std])
        
        # 4. Create a summary CSV with overall and average scores side by side for easy comparison
        with open(os.path.join(csv_dir, 'inception_score_summary.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Task', 'Round', 'Overall_Score', 'Overall_Std', 'Avg_Client_Score', 'Avg_Client_Std', 'Score_Difference'])
            
            # Write data
            for task in tasks:
                if task not in self.avg_inception_scores:
                    continue
                    
                rounds = sorted(set(self.inception_scores[task].keys()) & set(self.avg_inception_scores[task].keys()))
                for r in rounds:
                    overall_score = self.inception_scores[task][r]['overall_score']
                    overall_std = self.inception_scores[task][r]['overall_std']
                    avg_score = self.avg_inception_scores[task][r]['avg_score']
                    avg_std = self.avg_inception_scores[task][r]['avg_std']
                    
                    # Calculate difference between overall and average scores
                    score_diff = overall_score - avg_score
                    
                    writer.writerow([
                        task, r, overall_score, overall_std, avg_score, avg_std, score_diff
                    ])
        
        logger.info(f"Inception scores saved to CSV files in {csv_dir}")

    def aggregate_parameters_(self, class_partial):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples # length of the train data for weighted importance
        
        param_weight_sum = {}
        for user in self.selected_users:
            for name, param in user.model.named_parameters():
                if ('fc_classifier' in name and class_partial):
                    class_available = torch.Tensor(user.classes_so_far).long()
                    param_dict[name][class_available] += param.data[class_available] * user.train_samples / total_train
                    
                    add_weight = torch.zeros([param.data.shape[0]]).cuda()
                    add_weight[class_available] = user.train_samples / total_train
                else:
                    param_dict[name] += param.data * user.train_samples / total_train
                    add_weight = user.train_samples / total_train
                
                if name not in param_weight_sum.keys():
                    param_weight_sum[name] = add_weight
                else:
                    param_weight_sum[name] += add_weight
                
        for name, param in self.model.named_parameters():

            if 'fc_classifier' in name and class_partial:
                valid_class = (param_weight_sum[name]>0)
                weight_sum = param_weight_sum[name][valid_class]
                if 'weight' in name:
                    weight_sum = weight_sum.view(-1, 1)
                param.data[valid_class] = param_dict[name][valid_class]/weight_sum
            else:
                param.data = param_dict[name]/param_weight_sum[name]

    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_shared_parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio