import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_sizes=[128, 64, 32]):
        super(DNN, self).__init__()
        
        layers = []
        input_size = n_features
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, n_classes))
        
        self.network = nn.Sequential(*layers)
        self.n_classes = n_classes
    
    def forward(self, x):
        return self.network(x)
