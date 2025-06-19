import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def load_normalized_data(csv_path, n_collocation=10000, split_ratio=0.8, device="cpu"):
    """
    Loads CSV created from NYT dataset, splits into boundary & collocation sets,
    and returns torch tensors for training PINN.

    Args:
        csv_path (str): Path to preprocessed CSV file.
        n_collocation (int): Number of collocation points.
        split_ratio (float): Proportion of data used for training.
        device (str): 'cpu' or 'cuda'

    Returns:
        X_u_train: Tensor of coordinates with known u (for supervised loss)
        u_train: Corresponding known u values
        X_f: Collocation points (no observed u)
        X_test, u_test: Held-out evaluation set
    """

    print(f"Loading and normalizing dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"lon", "lat", "t", "u"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Normalize columns
    df["t"] = (df["t"] - df["t"].min()) / (df["t"].max() - df["t"].min())
    df["lon"] = (df["lon"] - df["lon"].min()) / (df["lon"].max() - df["lon"].min())
    df["lat"] = (df["lat"] - df["lat"].min()) / (df["lat"].max() - df["lat"].min())

    df["u"] = np.log1p(df["u"])
    df["u"] = df["u"] / df["u"].max()

    # Prepare arrays
    X = df[["lon", "lat", "t"]].to_numpy(dtype=np.float32)
    u = df["u"].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Split into train/test
    X_train, X_test, u_train, u_test = train_test_split(X, u, test_size=1 - split_ratio, random_state=42)

    # Generate random collocation points uniformly in [0,1]^3
    X_f = np.random.rand(n_collocation, 3).astype(np.float32)

    # Move everything to device
    X_u_train = torch.tensor(X_train, device=device)
    u_train = torch.tensor(u_train, device=device)
    X_f = torch.tensor(X_f, device=device)
    X_test = torch.tensor(X_test, device=device)
    u_test = torch.tensor(u_test, device=device)

    print(f"Loaded and processed: {len(df)} samples")
    print(f"Training points: {len(X_u_train)} | Test points: {len(X_test)} | Collocation points: {len(X_f)}")

    return X_u_train, u_train, X_f, X_test, u_test
