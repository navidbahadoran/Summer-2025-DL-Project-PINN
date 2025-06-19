import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch


# 1. Plot u(x, t) surface or heatmap
def plot_solution_surface(x, t, u_grid, title="Solution u(x,t)", method="FDM", cmap="viridis"):
    """
    x: 1D spatial grid (length Nx)
    t: 1D temporal grid (length Nt)
    u_grid: 2D array of shape (Nt, Nx)
    """
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(10, 5))
    plt.contourf(X, T, u_grid, 100, cmap=cmap)
    plt.title(f"{method} - {title}")
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar(label="u(x,t)")
    plt.tight_layout()
    plt.show()


# 2. Plot final-time comparison
def plot_final_time(x, u_fd, u_pinn, T):
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_fd, label="FDM")
    plt.plot(x, u_pinn, '--', label="PINN")
    plt.title(f"u(x, T={T})")
    plt.xlabel("x"); plt.ylabel("u(x,T)")
    plt.legend(); plt.grid(True)
    plt.show()


# 3. Plot pointwise error
def plot_error_surface(x, t, error_grid, title="Pointwise Error |u_PINN - u_FDM|", cmap="magma"):
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(10, 5))
    plt.contourf(X, T, error_grid, 100, cmap=cmap)
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("t")
    plt.colorbar(label="Error Magnitude")
    plt.tight_layout()
    plt.show()


# 4. Error metrics
def compute_errors(u_pred, u_true):
    diff = u_pred - u_true
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(u_true)
    max_err = np.max(np.abs(diff))
    return {
        "Relative L2 Error": rel_l2,
        "Max Abs Error": max_err
    }


# 5. Tabulate and display errors
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


def compute_rmse(y_pred, y_true, apply_expm1=True):
    """
    Compute Root Mean Squared Error (RMSE).
    If `apply_expm1` is True, convert log1p-transformed values back to original scale.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach()

    if apply_expm1:
        y_pred = torch.expm1(y_pred)
        y_true = torch.expm1(y_true)

    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()

    return np.sqrt(mean_squared_error(y_true_np, y_pred_np))


def plot_prediction_surface(X, u_pred, title="Predicted Surface"):
    """
    3D surface plot of predicted u(x, y, t).
    Applies inverse log1p to bring prediction to original scale.
    """
    if isinstance(u_pred, torch.Tensor):
        u_plot = torch.expm1(u_pred).cpu().numpy()
    else:
        u_plot = np.expm1(u_pred)

    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = X

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
    """
    Plots training loss over epochs.
    """
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
    """
    Plots time series of u for a specific (lon, lat) point.
    """
    df_sub = df[(df["lon"] == lon) & (df["lat"] == lat)].sort_values(by="t")
    plt.figure(figsize=(10, 4))
    plt.plot(df_sub["t"], np.expm1(df_sub["u"]), marker="o")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Normalized Cases (u)")
    plt.title(f"{title} at lon={lon:.3f}, lat={lat:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
