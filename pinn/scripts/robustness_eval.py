# scripts/robustness_eval.py (Refactored)

# scripts/robustness_eval.py (with PDE residual inspection added)

import torch
import numpy as np
from pinn.models.neural_net import PINN
from pinn.data.data_generator import load_normalized_data
from pinn.models.pinn_solver import PINNSolver
from pinn.scripts.adversarial import (
    perturb_initial_conditions, 
    perturb_boundary_conditions, 
    perturb_collocation_points
)
from utils import compute_metrics, compute_errors
from config import config
import matplotlib.pyplot as plt
import pickle
import os


def split_ic_bc(X_u, u, t_threshold=0.01):
    ic_mask = X_u[:, 2] < t_threshold
    bc_mask = ~ic_mask
    return X_u[ic_mask], u[ic_mask], X_u[bc_mask], u[bc_mask]


def compute_residual(model, X_f, beta=1.0):
    """
    Compute PDE residual f = u_t - Δu - βu on collocation points.
    Returns residual values, mean and max residual norms.
    """
    X_f = X_f.clone().detach().requires_grad_(True)
    u_pred = model(X_f)

    grads = torch.autograd.grad(
        outputs=u_pred,
        inputs=X_f,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True
    )[0]

    u_t = grads[:, 2:3]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, X_f, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X_f, grad_outputs=torch.ones_like(u_y),
                               create_graph=True, retain_graph=True)[0][:, 1:2]

    residual = u_t - (u_xx + u_yy) - beta * u_pred
    abs_residual = torch.abs(residual).detach().cpu().numpy()

    return abs_residual, np.mean(abs_residual), np.max(abs_residual)


def plot_residual_heatmap(X_f, abs_residual, title="|PDE Residual f(x,y,t)|"):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X_np = X_f.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=abs_residual.ravel(), cmap="inferno", s=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def run_robustness_sweep(epsilons=[0.0, 0.01, 0.02, 0.05, 0.1], epochs=2000):
    results = {}
    device = torch.device(config["device"])
    csv_path = config['data_path']

    for eps in epsilons:
        print(f"\n[INFO] Evaluating at epsilon = {eps:.3f}")

        X_u_train, u_train, X_f_train, X_test, u_test = load_normalized_data(
            csv_path=csv_path,
            n_collocation=config["n_collocation"],
            split_ratio=config["train_test_split"],
            device=device
        )

        X_ic, u_ic, X_bc, u_bc = split_ic_bc(X_u_train, u_train)
        X_ic, u_ic = perturb_initial_conditions(X_ic, u_ic, epsilon=eps)
        X_bc, u_bc = perturb_boundary_conditions(X_bc, u_bc, epsilon=eps)
        X_f_train = perturb_collocation_points(X_f_train, scale=eps)

        model = PINN(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            hidden_layers=config["hidden_layers"],
            output_dim=config["output_dim"]
        ).to(device)

        solver = PINNSolver(model, X_u_train, u_train, X_f_train,
                            X_ic=X_ic, u_ic=u_ic, X_bc=X_bc, u_bc=u_bc,
                            device=device)

        solver.train(epochs=epochs, lr=config["learning_rate"])

        model.eval()
        with torch.no_grad():
            u_pred = model(X_test)
            metrics = compute_metrics(u_pred, u_test, apply_expm1=True)
            errors = compute_errors(u_pred.cpu().numpy(), u_test.cpu().numpy())

        abs_residual, mean_res, max_res = compute_residual(model, X_f_train)
        print(f"[ε={eps:.2f}] RMSE = {metrics['rmse']:.4f} | Rel L2 = {errors['Relative L2 Error']:.4f} | Max Error = {errors['Max Abs Error']:.4f}")
        print(f"PDE Residuals: Mean = {mean_res:.4e}, Max = {max_res:.4e}")

        results[eps] = {
            "rmse": metrics["rmse"],
            **errors,
            "loss_history": solver.loss_history,
            "mean_residual": mean_res,
            "max_residual": max_res
        }

        if eps == 0.0:
            plot_residual_heatmap(X_f_train, abs_residual, title="PDE Residual Heatmap (ε=0.0)")

    return results


def plot_robustness_curve(results):
    eps_vals = list(results.keys())
    rmses = [results[eps]["rmse"] for eps in eps_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(eps_vals, rmses, marker='o', linestyle='-', color='darkblue')
    plt.title("Robustness Curve: PINN vs Adversarial Perturbations")
    plt.xlabel("Perturbation Magnitude (ε)")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sweep_results = run_robustness_sweep()
    plot_robustness_curve(sweep_results)

    with open("checkpoints/robustness_results.pkl", "wb") as f:
        pickle.dump(sweep_results, f)
    print("[INFO] Saved sweep results to checkpoints/robustness_results.pkl")
