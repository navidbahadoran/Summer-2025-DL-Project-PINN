import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from config import config


def split_ic_bc(X_u, u, t_threshold=0.01, eps=1e-3):
    # Initial Condition: t ≈ 0
    ic_mask = X_u[:, 2] < t_threshold

    # Boundary Condition: x or y is near boundary (at any time)
    x_min, x_max = X_u[:, 0].min(), X_u[:, 0].max()
    y_min, y_max = X_u[:, 1].min(), X_u[:, 1].max()
    x = X_u[:, 0]
    y = X_u[:, 1]
    
    bc_mask = (
        ((x - x_min).abs() < eps) |
        ((x_max - x).abs() < eps) |
        ((y - y_min).abs() < eps) |
        ((y_max - y).abs() < eps)
    ) & (~ic_mask)  # Make sure IC and BC are exclusive

    # Return IC and BC subsets
    return X_u[ic_mask], u[ic_mask], X_u[bc_mask], u[bc_mask]


def load_normalized_data(csv_path, n_collocation=10000, split_ratio=0.8, device="cpu"):
    """
    Loads CSV created from NYT dataset, splits into boundary & collocation sets,
    and returns torch tensors for training PINN.
    """

    print(f"Loading and normalizing dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"lon", "lat", "t", "u"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # === Apply log transform to u ===
    df["u"] = np.log1p(df["u"].values)

    # === Prepare inputs and targets ===
    X = df[["lon", "lat", "t"]].values.astype(np.float32)
    u = df["u"].values.reshape(-1, 1).astype(np.float32)

    # === Fit scalers ===
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    u_scaled = y_scaler.fit_transform(u)

    # === Save scalers ===
    torch.save(y_scaler, config["pinn_scaler_y"])
    torch.save(x_scaler, config["pinn_scaler_x"])

    # === Split supervised data ===
    X_train, X_test, u_train, u_test = train_test_split(X_scaled, u_scaled, test_size=1 - split_ratio, random_state=42)

    # === Generate collocation points ===
    X_f = np.random.rand(n_collocation, 3).astype(np.float32)

    # === Convert to torch tensors ===
    X_u_train = torch.tensor(X_train, device=device)
    u_train = torch.tensor(u_train, device=device)
    X_f = torch.tensor(X_f, device=device)
    X_test = torch.tensor(X_test, device=device)
    u_test = torch.tensor(u_test, device=device)

    print(f"[INFO] Data summary:")
    print(f"  → Train points: {len(X_u_train)}")
    print(f"  → Test points : {len(X_test)}")
    print(f"  → Collocation : {len(X_f)}")

    return X_u_train, u_train, X_f, X_test, u_test