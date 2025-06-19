# models/neural_net.py

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, hidden_layers=4, output_dim=1):
        """
        Physics-Informed Neural Network architecture.
        
        Parameters:
        - input_dim:  Number of input features (x, y, t)
        - hidden_dim: Number of neurons in hidden layers
        - hidden_layers: Total number of hidden layers
        - output_dim: Output dimension (u)
        """
        super(PINN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Assemble into model
        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Xavier weight initialization to help with stable training.
        """
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.model(x)
