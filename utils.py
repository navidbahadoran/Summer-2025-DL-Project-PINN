# utils.py — Unified Utilities for PINN and GNN
# ============================================

# ----------- Shared Imports ----------- #
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ----------- Shared Utilities ----------- #
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# ----------- PINN-Specific Utilities ----------- #
def plot_solution_surface(x, t, u_grid, title="Solution u(x,t)", method="FDM", cmap="viridis"):
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(10, 5))
    plt.contourf(X, T, u_grid, 100, cmap=cmap)
    plt.title(f"{method} - {title}")
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar(label="u(x,t)")
    plt.tight_layout()
    plt.show()


def plot_final_time(x, u_fd, u_pinn, T):
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_fd, label="FDM")
    plt.plot(x, u_pinn, '--', label="PINN")
    plt.title(f"u(x, T={T})")
    plt.xlabel("x"); plt.ylabel("u(x,T)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_surface(X, Y, error_grid, title="Error Surface", cmap="magma"):
    plt.figure(figsize=(10, 5))
    cp = plt.pcolormesh(X, Y, error_grid, cmap=cmap, shading="auto")
    plt.colorbar(cp, label="Error Magnitude")
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("t")
    plt.tight_layout()
    plt.show()


def compute_errors(u_pred, u_true):
    diff = u_pred - u_true
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(u_true)
    max_err = np.max(np.abs(diff))
    return {
        "Relative L2 Error": rel_l2,
        "Max Abs Error": max_err
    }


def error_table(metrics_dict):
    df = pd.DataFrame({
        "Metric": list(metrics_dict.keys()),
        "Value": [f"{v:.2e}" for v in metrics_dict.values()]
    })
    return df


def plot_loss_history(loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Training Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_metrics(y_pred, y_true, apply_expm1=True):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if apply_expm1:
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae}


def plot_prediction_surface(X, u_pred, title="Predicted Surface"):
    u_plot = torch.expm1(u_pred).cpu().numpy() if isinstance(u_pred, torch.Tensor) else np.expm1(u_pred)
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(X_np[:, 0], X_np[:, 1], u_plot.ravel(), cmap="viridis", linewidth=0.1)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Normalized Cases (u)")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_time_series(df, lon, lat, title="Time Series at Location"):
    df_sub = df[(df["lon"] == lon) & (df["lat"] == lat)].sort_values(by="t")
    plt.figure(figsize=(10, 4))
    plt.plot(df_sub["t"], np.expm1(df_sub["u"]), marker="o")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Normalized Cases (u)")
    plt.title(f"{title} at lon={lon:.3f}, lat={lat:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# # ----------- GNN-Specific Utilities ----------- #
# def compute_metrics(preds, targets):
#     rmse = np.sqrt(mean_squared_error(targets, preds))
#     mae = mean_absolute_error(targets, preds)
#     return {'rmse': rmse, 'mae': mae}


def plot_learning_curve(train_losses, val_rmse=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    if val_rmse is not None:
        plt.plot(val_rmse, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("GNN Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_truth(preds, targets, sample_size=200):
    indices = np.random.choice(len(preds), min(sample_size, len(preds)), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(targets[indices], preds[indices], alpha=0.7)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Ground Truth")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_gnn(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).squeeze()
            preds.append(out.cpu().numpy())
            targets.append(data.y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return compute_metrics(preds, targets), preds, targets
