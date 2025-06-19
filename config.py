import torch

config = {
    # Data
    "data_path": "D:/Programming/Summer-2025-DL-Project-PINN/data/covid_county_cases.csv",  # Path to preprocessed CSV
    "train_test_split": 0.8,                     # Ratio for train/test split
    "n_collocation": 10000,                      # Number of collocation (physics) points

    # Model
    "input_dim": 3,       # (x, y, t)
    "output_dim": 1,      # u
    "hidden_dim": 64,     # Neurons per hidden layer
    "hidden_layers": 4,   # Number of hidden layers

    # Training
    "epochs": 5000,
    "learning_rate": 1e-3,
    "batch_size": None,   # Full-batch training (can set to mini-batch later)
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Logging and Evaluation
    "print_every": 500,                         # Print frequency
    "checkpoint_path": "D:/Programming/Summer-2025-DL-Project-PINN/checkpoints/model.pth", # Model save path (optional)
    "plot_results": True                        # Toggle result plotting
}
