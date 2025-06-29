import torch

config = {
    # Data
    "data_path": "./dataset/covid_county_cases.csv",  # Path to preprocessed CSV
    "gnn_dataset_path": "./dataset/processed_graph_dataset.pt",  # Path to gnn preprocessed CSV
    "fd_path": "./dataset/fd_solution.npz",
    "zip_path": "./dataset/county_shapefiles.zip",
    "extract_path": "./dataset/county_shapefiles",
    "shape_file_path": "./dataset/county_shapefiles/cb_2022_us_county_20m.shp",
    "loss_path": "./checkpoints/loss_history.pkl",

    "train_test_split": 0.8,                     # Ratio for train/test split
    "n_collocation": 10000,                      # Number of collocation (physics) points

   # Model Architecture
    "input_dim": 3,                                  # (x, y, t)
    "output_dim": 1,                                 # u
    "hidden_dim": 64,                                # Neurons per hidden layer
    "hidden_layers": 4,                              # Number of hidden layers
    "use_batch_norm": True,                          # Batch normalization toggle
    "activation": "tanh",                            # Activation function
    "initializer": "xavier",                         # Weight initializer
    # GNN model configuration
    "gnn_hidden_dim": 64,       # Number of hidden units
    "gnn_in_dim": 3,            # Input features per node (e.g., [cases, lat, lon])
    "gnn_out_dim": 1,           # Output dimension (e.g., predicted case count)


    # Training
    "epochs": 2000,
    "learning_rate": 1e-3,
    "batch_size": 4,   # Full-batch training (can set to mini-batch later)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 50,

    # Logging and Evaluation
    "print_interval": 500,                         # Print frequency
    "checkpoint_path": "./checkpoints/model.pth", # Model save path
    "gnn_model_path": "./checkpoints/gnn_model.pth", # gnn Model save path
    "gnn_scaler_y": "./checkpoints/gnn_target_scaler.pt",
    "gnn_scaler_x": "./checkpoints/gnn_input_scaler.pt",
    "plot_results": True                        # Toggle result plotting
}
