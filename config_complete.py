
import torch

config = {
    # Data
    "data_path": "D:/Programming/Summer-2025-DL-Project-PINN/data/covid_county_cases.csv",
    "train_test_split": 0.8,
    "n_collocation": 10000,

    # Model
    "input_dim": 3,
    "output_dim": 1,
    "hidden_dim": 64,
    "hidden_layers": 4,
    "use_batch_norm": True,
    "activation": "tanh",         # Supported: tanh, relu, gelu, sigmoid
    "initializer": "xavier",      # Supported: xavier, kaiming, normal

    # Training
    "epochs": 5000,
    "learning_rate": 1e-3,
    "batch_size": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Logging and Evaluation
    "print_every": 500,
    "checkpoint_path": "D:/Programming/Summer-2025-DL-Project-PINN/checkpoints/model.pth",
    "plot_results": True
}
