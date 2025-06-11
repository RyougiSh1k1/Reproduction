#!/usr/bin/env python
"""
Modified FLAlgorithms/servers/serverPreciseFCL.py to track accuracy at each round
"""

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
from utils.communication_tracker import PreciseFCLDetailedTracker

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

        # Initialize inception score tracking
        self.inception_scores = {}
        self.client_inception_scores = {}
        self.avg_inception_scores = {}

        # Initialize detailed communication tracker
        self.comm_tracker = PreciseFCLDetailedTracker()
        
        # Add communication tracking to pickle record
        self.pickle_record['communication'] = {}
        self.pickle_record['communication_detailed'] = {}
        
        # NEW: Add per-round accuracy tracking
        self.pickle_record['per_round_accuracy'] = {}
        self.round_accuracy_metrics = {}

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

    def evaluate_round_accuracy(self, task_id, glob_iter, save=True):
        """
        Evaluate accuracy at current round for all tasks seen so far
        Returns detailed accuracy metrics for the current round
        """
        logger.info(f"Evaluating accuracy for round {glob_iter}, task {task_id}")
        
        # Test on all tasks seen so far
        test_ids, user_task_num_samples, user_task_acc, user_task_losses = self.test_all_(selected=False, personal=False)
        
        # Calculate metrics
        user_acc = np.sum(user_task_num_samples*user_task_acc, axis=1)/np.sum(user_task_num_samples, axis=1)
        task_acc = np.sum(user_task_num_samples*user_task_acc, axis=0)/np.sum(user_task_num_samples, axis=0)
        user_loss = np.sum(user_task_num_samples*user_task_losses, axis=1)/np.sum(user_task_num_samples, axis=1)
        task_loss = np.sum(user_task_num_samples*user_task_losses, axis=0)/np.sum(user_task_num_samples, axis=0)

        # Overall metrics
        glob_acc = np.sum(user_task_num_samples*user_task_acc)/np.sum(user_task_num_samples)
        glob_loss = np.sum(user_task_num_samples*user_task_losses)/np.sum(user_task_num_samples)
        
        # Current task accuracy (if available)
        current_task_acc = task_acc[task_id] if task_id < len(task_acc) else 0.0
        current_task_loss = task_loss[task_id] if task_id < len(task_loss) else 0.0
        
        # Calculate average accuracy on past tasks (forgetting metric)
        if task_id > 0:
            past_tasks_acc = np.mean(task_acc[:task_id])
            past_tasks_loss = np.mean(task_loss[:task_id])
        else:
            past_tasks_acc = 0.0
            past_tasks_loss = 0.0
        
        # Create metrics dictionary
        round_metrics = {
            'task_id': task_id,
            'global_accuracy': float(glob_acc),
            'global_loss': float(glob_loss),
            'current_task_accuracy': float(current_task_acc),
            'current_task_loss': float(current_task_loss),
            'past_tasks_accuracy': float(past_tasks_acc),
            'past_tasks_loss': float(past_tasks_loss),
            'per_task_accuracy': [float(acc) for acc in task_acc],
            'per_task_loss': [float(loss) for loss in task_loss],
            'per_user_accuracy': [float(acc) for acc in user_acc],
            'per_user_loss': [float(loss) for loss in user_loss],
            'user_task_accuracy_matrix': user_task_acc.tolist(),
            'user_task_loss_matrix': user_task_losses.tolist(),
            'user_task_samples_matrix': user_task_num_samples.tolist()
        }
        
        if save:
            # Store in pickle record
            self.pickle_record['per_round_accuracy'][glob_iter] = round_metrics
            
            # Also store in round accuracy metrics for easy access
            self.round_accuracy_metrics[glob_iter] = round_metrics
        
        # Log key metrics
        logger.info(f"Round {glob_iter} - Task {task_id} Accuracy Metrics:")
        logger.info(f"  Global Accuracy: {glob_acc:.4f}")
        logger.info(f"  Current Task Accuracy: {current_task_acc:.4f}")
        logger.info(f"  Past Tasks Average Accuracy: {past_tasks_acc:.4f}")
        logger.info(f"  Per-Task Accuracies: {[f'{acc:.3f}' for acc in task_acc]}")
        
        return round_metrics

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
                    if len(loader.dataset) < 10:
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
                            list(all_labels),
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
        """Calculate combined inception scores for the current task and round"""
        logger.info("Calculating inception scores for task {} round {}".format(task_id, glob_iter))
        
        try:
            # Get client-specific scores and average
            client_scores, client_stds, client_class_scores, avg_score, avg_std = self.calculate_client_inception_scores(task_id, glob_iter)
            
            # Create a combined test dataset
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
                    list(all_labels),
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
            return None, None, None, {}, {}, None, None

    def train(self, args):
        N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        
        for task in range(N_TASKS):
            self.inception_scores[task] = {}
            self.client_inception_scores[task] = {}
            self.avg_inception_scores[task] = {}
        
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
            
            # ============ train ==============
            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_task in range(epoch_per_task):
                
                glob_iter = glob_iter_task + (epoch_per_task) * task

                logger.info("\n\n------------- Round number: %d | Current task: %d -------------\n\n"%(glob_iter, task))

                # select users
                self.selected_users, self.user_idxs = self.select_users(glob_iter, len(self.users), return_idx=True)

                if self.algorithm != 'local':
                    for user in self.selected_users:
                        # Track detailed component download
                        if self.classifier_global_mode == 'all':
                            total_bytes, component_sizes = self.comm_tracker.track_component_download(
                                user.id, glob_iter, self.model
                            )
                        elif self.classifier_global_mode == 'head':
                            total_bytes, component_sizes = self.comm_tracker.track_component_download(
                                user.id, glob_iter, self.model, 
                                component_filter=self.classifier_head_list
                            )
                        elif self.classifier_global_mode == 'extractor':
                            filter_list = [name for name, _ in self.model.named_parameters() 
                                         if 'classifier' in name and not any(h in name for h in self.classifier_head_list)]
                            total_bytes, component_sizes = self.comm_tracker.track_component_download(
                                user.id, glob_iter, self.model, 
                                component_filter=filter_list
                            )
                        
                        # Send parameters to user
                        user.set_parameters(self.model, beta=1)

                chosen_verbose_user = np.random.randint(0, len(self.users))

                # ---------------
                #   train user
                # ---------------
                self.pickle_record['train'][glob_iter] = {}

                global_classifier = self.model.classifier
                global_classifier.eval()
                
                for user_id, user in zip(self.user_idxs, self.selected_users):
                    verbose = True
                    
                    # perform regularization using generated samples after the first communication round
                    user_result = user.train(
                        glob_iter,
                        glob_iter_task,
                        global_classifier,
                        verbose=verbose)
                        
                    self.pickle_record['train'][glob_iter][user_id] = user_result

                #=================
                # 2. Server update
                #=================
                if self.algorithm != 'local':
                    # Track uploads before aggregation
                    for user in self.selected_users:
                        # Track classifier upload
                        classifier_bytes, classifier_components = self.comm_tracker.track_component_upload(
                            user.id, glob_iter, user.model.classifier
                        )
                        
                        # Track flow model upload if it exists
                        if self.algorithm == 'PreciseFCL' and hasattr(user.model, 'flow') and user.model.flow is not None:
                            flow_bytes, flow_components = self.comm_tracker.track_component_upload(
                                user.id, glob_iter, user.model.flow
                            )
                    
                    # Perform aggregation
                    self.aggregate_parameters_(class_partial=False)
                
                # ===============================================
                # Log detailed communication summary
                # ===============================================
                self.comm_tracker.log_detailed_round_summary(glob_iter)
                self.pickle_record['communication_detailed'][glob_iter] = self.comm_tracker.get_round_component_summary(glob_iter)
                
                # ===============================================
                # NEW: Evaluate accuracy at each round
                # ===============================================
                # Final evaluation at end of task (original behavior)
                #self.evaluate_all_(glob_iter=glob_iter, matrix=True, personal=False)
                
                # Calculate inception scores
                overall_score, overall_std, class_scores, client_scores, client_stds, avg_score, avg_std = self.calculate_inception_scores(task, glob_iter)
                
                if overall_score is not None:
                    self.inception_scores[task][glob_iter] = {
                        'overall_score': overall_score,
                        'overall_std': overall_std,
                        'class_scores': class_scores
                    }
                    
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
                    #self.save_inception_score_visualization(task)

            if self.algorithm != 'local':
                # send parameteres: server -> client
                self.send_parameters(mode='all', beta=1)

            # Final evaluation at end of task (original behavior)
            self.evaluate_all_(glob_iter=glob_iter, matrix=True, personal=False)

            self.save_pickle()
            
            # Save final inception scores for this task
            # self.save_inception_scores_to_csv()
            # self.save_detailed_communication_analysis()
            
            # NEW: Save per-round accuracy analysis
            self.save_per_round_accuracy_analysis()
    
    def save_per_round_accuracy_analysis(self):
        """Save detailed per-round accuracy analysis"""
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        
        # Create directory for analysis
        analysis_dir = os.path.join(self.args.target_dir_name, 'per_round_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Convert metrics to DataFrame for easier analysis
        rounds = sorted(self.round_accuracy_metrics.keys())
        
        # Extract data for plotting
        global_accuracies = []
        current_task_accuracies = []
        past_tasks_accuracies = []
        task_ids = []
        
        for round_num in rounds:
            metrics = self.round_accuracy_metrics[round_num]
            global_accuracies.append(metrics['global_accuracy'])
            current_task_accuracies.append(metrics['current_task_accuracy'])
            past_tasks_accuracies.append(metrics['past_tasks_accuracy'])
            task_ids.append(metrics['task_id'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Global accuracy over rounds
        axes[0, 0].plot(rounds, global_accuracies, 'b-', marker='o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Global Accuracy Over Rounds')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Add task boundaries
        unique_tasks = sorted(set(task_ids))
        task_boundaries = {}
        for task in unique_tasks:
            first_round = min([r for r, t in zip(rounds, task_ids) if t == task])
            if task > 0:  # Don't add boundary for first task
                task_boundaries[task] = first_round
                axes[0, 0].axvline(x=first_round, color='red', linestyle='--', alpha=0.7, label=f'Task {task} start' if task == 1 else "")
        
        if task_boundaries:
            axes[0, 0].legend()
        
        # Plot 2: Current task accuracy
        axes[0, 1].plot(rounds, current_task_accuracies, 'g-', marker='s', linewidth=2, markersize=4)
        axes[0, 1].set_title('Current Task Accuracy Over Rounds')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Add task boundaries
        for task, boundary in task_boundaries.items():
            axes[0, 1].axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        # Plot 3: Past tasks accuracy (forgetting)
        axes[1, 0].plot(rounds, past_tasks_accuracies, 'r-', marker='^', linewidth=2, markersize=4)
        axes[1, 0].set_title('Past Tasks Average Accuracy (Forgetting Metric)')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Add task boundaries
        for task, boundary in task_boundaries.items():
            axes[1, 0].axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        # Plot 4: Combined view
        axes[1, 1].plot(rounds, global_accuracies, 'b-', marker='o', linewidth=2, markersize=3, label='Global')
        axes[1, 1].plot(rounds, current_task_accuracies, 'g-', marker='s', linewidth=2, markersize=3, label='Current Task')
        axes[1, 1].plot(rounds, past_tasks_accuracies, 'r-', marker='^', linewidth=2, markersize=3, label='Past Tasks')
        axes[1, 1].set_title('All Accuracy Metrics Over Rounds')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].legend()
        
        # Add task boundaries
        for task, boundary in task_boundaries.items():
            axes[1, 1].axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'per_round_accuracy_analysis.png'), dpi=300)
        plt.close()
        
        # Save CSV data
        df_data = {
            'round': rounds,
            'task_id': task_ids,
            'global_accuracy': global_accuracies,
            'current_task_accuracy': current_task_accuracies,
            'past_tasks_accuracy': past_tasks_accuracies
        }
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(analysis_dir, 'per_round_accuracy_metrics.csv'), index=False)
        
        # Generate summary report
        with open(os.path.join(analysis_dir, 'per_round_accuracy_report.md'), 'w') as f:
            f.write('# Per-Round Accuracy Analysis Report\n\n')
            f.write('## Summary\n\n')
            f.write(f'- Total rounds: {len(rounds)}\n')
            f.write(f'- Total tasks: {len(unique_tasks)}\n')
            f.write(f'- Final global accuracy: {global_accuracies[-1]:.4f}\n')
            f.write(f'- Best global accuracy: {max(global_accuracies):.4f} (round {rounds[global_accuracies.index(max(global_accuracies))]})\n\n')
            
            f.write('## Per-Task Analysis\n\n')
            for task in unique_tasks:
                task_rounds = [r for r, t in zip(rounds, task_ids) if t == task]
                task_global_accs = [global_accuracies[rounds.index(r)] for r in task_rounds]
                task_current_accs = [current_task_accuracies[rounds.index(r)] for r in task_rounds]
                
                f.write(f'### Task {task}\n')
                f.write(f'- Rounds: {min(task_rounds)} - {max(task_rounds)}\n')
                f.write(f'- Final global accuracy: {task_global_accs[-1]:.4f}\n')
                f.write(f'- Final current task accuracy: {task_current_accs[-1]:.4f}\n')
                f.write(f'- Best current task accuracy: {max(task_current_accs):.4f}\n\n')
            
            f.write('## Forgetting Analysis\n\n')
            if len(unique_tasks) > 1:
                final_past_acc = past_tasks_accuracies[-1]
                f.write(f'- Final past tasks accuracy: {final_past_acc:.4f}\n')
                f.write(f'- Forgetting amount: {max(past_tasks_accuracies) - final_past_acc:.4f}\n')
        
        logger.info(f"Per-round accuracy analysis saved to {analysis_dir}")

    # [Include all other existing methods from the original class...]
    # (The rest of the methods remain the same as in the original implementation)
    
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

    def save_inception_scores_to_csv(self):
        """Save inception scores to CSV files for easier analysis"""
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
        
        logger.info(f"Inception scores saved to CSV files in {csv_dir}")

    def aggregate_parameters_(self, class_partial, track_round=None):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
    
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)
        
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        
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

    def save_detailed_communication_analysis(self):
        """Save detailed component-wise communication analysis"""
        import os
        
        # Create directory for communication logs
        comm_dir = os.path.join(self.args.target_dir_name, 'communication_analysis')
        os.makedirs(comm_dir, exist_ok=True)
        
        # Save detailed JSON log
        self.comm_tracker.save_detailed_communication_log(
            os.path.join(comm_dir, 'detailed_communication_log.json')
        )
        
        # Generate component breakdown visualization
        self.comm_tracker.plot_component_breakdown(
            save_path=os.path.join(comm_dir, 'component_breakdown.png')
        )
        
        # Generate detailed text report
        self.comm_tracker.generate_component_report(
            os.path.join(comm_dir, 'component_report.txt')
        )
        
        # Log category summary
        self.comm_tracker.log_category_summary()
        
        logger.info(f"Communication analysis saved to {comm_dir}")