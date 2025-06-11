import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import glog as logger

def inception_score(imgs, batch_size=32, resize=True, splits=10, device='cuda'):
    """Calculate the inception score of the generated images"""
    N = len(imgs)
    
    # Set up inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get predictions
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = imgs[i:i+batch_size]
            if resize:
                batch = [preprocess(img) for img in batch]
            batch = torch.stack(batch, 0).to(device)
            
            # Get predictions
            pred = inception_model(batch)
            
            # If model output is logits, convert to probabilities
            if pred.shape[1] == 1000:
                pred = F.softmax(pred, dim=1)
            
            preds.append(pred.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    
    # Split predictions to calculate the score
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
        
    return np.mean(split_scores), np.std(split_scores)

class InceptionScoreCalculator:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
        # Initialize inception model
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def calculate(self, num_samples=1000, batch_size=32, splits=10):
        """Calculate the inception score for the given dataloader"""
        # Collect some samples
        samples = []
        target = []
        with torch.no_grad():
            for x, y in self.dataloader:
                samples.append(x)
                target.append(y)
                if len(samples) * x.size(0) >= num_samples:
                    break
                    
        # Concatenate samples and select the required number
        samples = torch.cat(samples, dim=0)[:num_samples]
        target = torch.cat(target, dim=0)[:num_samples]
        
        # Get predictions
        preds = []
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size].to(self.device)
                
                # Ensure the input is properly sized for Inception
                if batch.size(2) != 299 or batch.size(3) != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                pred = self.model(batch)
                
                # If model output is logits, convert to probabilities
                if pred.shape[1] == 1000:
                    pred = F.softmax(pred, dim=1)
                
                preds.append(pred.cpu().numpy())
                
        preds = np.concatenate(preds, axis=0)
        
        # Split predictions to calculate the score
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k+1) * (len(preds) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            
        return np.mean(split_scores), np.std(split_scores)

def calculate_inception_score_for_classes(dataloader, class_labels, device='cuda'):
    """Calculate inception scores for each class in the dataset"""
    calculator = InceptionScoreCalculator(dataloader, device)
    
    # Convert class_labels to a set for faster lookups
    class_labels_set = set(class_labels)
    
    # Group data by class
    class_samples = {label: [] for label in class_labels}
    class_targets = {label: [] for label in class_labels}
    
    # Keep track of unexpected labels
    unexpected_labels = set()
    
    with torch.no_grad():
        for x, y in dataloader:
            for i, label in enumerate(y):
                label_item = label.item()
                
                # Check if the label is in the expected class labels
                if label_item in class_labels_set:
                    class_samples[label_item].append(x[i:i+1])
                    class_targets[label_item].append(label_item)
                else:
                    # Keep track of unexpected labels
                    unexpected_labels.add(label_item)
    
    # Log any unexpected labels found
    if unexpected_labels:
        logger.warning(f"Found unexpected class labels: {unexpected_labels}. These will be ignored.")
    
    # Calculate inception score for each class
    class_scores = {}
    for label in class_labels:
        if len(class_samples[label]) > 10:  # Ensure we have enough samples
            samples = torch.cat(class_samples[label], dim=0)
            # Create a temporary dataloader for this class
            class_dataloader = DataLoader(
                [(samples[i], class_targets[label][i]) for i in range(len(samples))],
                batch_size=32,
                shuffle=False
            )
            score, std = calculator.calculate(num_samples=min(1000, len(samples)))
            class_scores[label] = (score, std)
        else:
            class_scores[label] = (0, 0)  # Not enough samples
    
    # Calculate overall inception score
    all_samples = []
    all_targets = []
    for label in class_labels:
        all_samples.extend(class_samples[label])
        all_targets.extend(class_targets[label])
    
    if len(all_samples) > 10:
        samples = torch.cat(all_samples, dim=0)
        # Create a temporary dataloader for all classes
        all_dataloader = DataLoader(
            [(samples[i], all_targets[i]) for i in range(len(samples))],
            batch_size=32,
            shuffle=False
        )
        overall_score, overall_std = calculator.calculate(num_samples=min(1000, len(samples)))
    else:
        overall_score, overall_std = 0, 0
    
    return overall_score, overall_std, class_scores