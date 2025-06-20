# FLAlgorithms/PreciseFCLNet/ili_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ILI_FCN(nn.Module):
    """Fully Connected Network for ILI time-series data"""
    def __init__(self, input_dim, hidden_dims, xa_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Feature extraction layer
        self.fc1 = nn.Linear(prev_dim, xa_dim)
        self.fc2 = nn.Linear(xa_dim, xa_dim)
        
        # Classifier
        self.fc_classifier = nn.Linear(xa_dim, num_classes)
        
        # Activation
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        xa = self.forward_to_xa(x)
        classes_p, logits = self.forward_from_xa(xa)
        return classes_p, xa, logits
    
    def forward_to_xa(self, x):
        # Ensure input is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        x = self.encoder(x)
        xa = F.relu(self.fc1(x))
        return xa
    
    def forward_from_xa(self, xa):
        xb = F.relu(self.fc2(xa))
        logits = self.fc_classifier(xb)
        classes_p = self.softmax(logits)
        return classes_p, logits