# models/neural_net.py

# models/neural_net.py

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, hidden_layers=4, output_dim=1, use_batch_norm=False):
        super(PINN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers with optional residuals
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def forward(self, x):
        return self.model(x)
