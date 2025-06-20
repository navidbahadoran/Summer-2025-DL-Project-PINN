# main_train.py

import os
import torch
from models.neural_net import PINN
from data.data_generator import load_normalized_data
from pinn.pinn_solver import PINNSolver
from utils import compute_rmse, plot_prediction_surface
from config import config
import pickle


def main():
    print("Starting PINN Training Pipeline")
    device = torch.device(config["device"])
    print(f"Using device: {device}\n")

    # ===========================
    # SECTION 1: Load the Dataset
    # ===========================
    print("Loading and preprocessing data...")
    X_u_train, u_train, X_f_train, X_test, u_test = load_normalized_data(
        csv_path=config["data_path"],
        n_collocation=config["n_collocation"],
        split_ratio=config["train_test_split"],
        device=device
    )

    # ===============================
    # SECTION 2: Initialize the Model
    # ===============================
    print("Initializing PINN architecture...")
    model = PINN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        hidden_layers=config["hidden_layers"],
        output_dim=config["output_dim"]
    ).to(device)

    # ===============================
    # SECTION 3: Train with PINNSolver
    # ===============================
    print("Starting training loop...")
    solver = PINNSolver(model, X_u_train, u_train, X_f_train, device=device)
    solver.train(epochs=config["epochs"], lr=config["learning_rate"])

    # ===========================
    # SECTION 4: Evaluate Results
    # ===========================
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test)
        u_pred_exp = torch.expm1(u_pred)
        u_test_exp = torch.expm1(u_test)
        rmse = compute_rmse(u_pred, u_test, apply_expm1=True)
        print(f"RMSE on test set: {rmse:.6f}")

    if config["plot_results"]:
        plot_prediction_surface(X_test, u_pred, title="Predicted u(x, y, t)")

    # ===========================
    # SECTION 5: Save the Model
    # ===========================
    checkpoint_path = config["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    with open("checkpoints/loss_history.pkl", "wb") as f:
    pickle.dump(solver.loss_history, f)
    print(f"Model checkpoint saved to: {checkpoint_path}\n")


if __name__ == "__main__":
    main()
