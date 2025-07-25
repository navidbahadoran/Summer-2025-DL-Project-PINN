import os
import pickle
import argparse
import torch
import numpy as np
from config import config
from utils import (plot_prediction_surface, save_model,
                    compute_metrics, set_seed, 
                    plot_prediction_vs_truth)
from pinn.models.pinn_solver import PINNSolver
from pinn.models.neural_net import PINN
from pinn.data.raw_data_processor import generate_processed_dataset
from pinn.scripts.adversarial import perturb_initial_conditions, perturb_boundary_conditions, perturb_collocation_points
from pinn.data.data_generator import load_normalized_data, split_ic_bc
from gnn.models.gnn_model import GNNPredictor
from gnn.data.graph_dataset import build_static_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib


def train_pinn():
    print("[INFO] Training PINN...")
    device = torch.device(config["device"])
    csv_path = config['data_path']
    generate_processed_dataset(output_path=csv_path, date_cutoff = None)

    X_u_train, u_train, X_f_train, X_test, u_test = load_normalized_data(
        csv_path, config["n_collocation"], config["train_test_split"], device)

    X_ic, u_ic, X_bc, u_bc = split_ic_bc(X_u_train, u_train,t_threshold=0.01, eps=1e-3)
    X_ic, u_ic = perturb_initial_conditions(X_ic, u_ic, epsilon=0.02)
    X_bc, u_bc = perturb_boundary_conditions(X_bc, u_bc, epsilon=0.02)
    X_f_train = perturb_collocation_points(X_f_train, scale=0.01)

    model = PINN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        hidden_layers=config["hidden_layers"],
        output_dim=config["output_dim"]
    ).to(device)
    
    solver = PINNSolver(model, X_u_train, u_train, X_f_train,
                        X_ic=X_ic, u_ic=u_ic, X_bc=X_bc, u_bc=u_bc, device=device)
    solver.train(config["epochs"], config["learning_rate"])

    model.load_state_dict(torch.load(config["pinn_model_path"], map_location=device))
    model.eval()

    with torch.no_grad():
        u_pred =model.predict(X_test)

    # === Inverse transform ===
    scaler_y_path = config["pinn_scaler_y"]
    if os.path.exists(scaler_y_path):
        scaler_y = torch.load(scaler_y_path, weights_only=False)
        # scaler_y = joblib.load(scaler_y_path)
        preds = scaler_y.inverse_transform(u_pred.cpu().numpy().reshape(-1, 1)).flatten()
        targets = scaler_y.inverse_transform(u_test.cpu().numpy().reshape(-1, 1)).flatten()
    else:
        print("[WARNING] Target scaler not found, skipping inverse transform.")
        preds = u_pred.cpu().numpy()
        targets = u_test.cpu().numpy()

    metrics = compute_metrics(preds, targets, apply_expm1=True)
    
    print(f"GNN RMSE (original scale): {metrics['rmse']:.4f}, MAE (original scale): {metrics['mae']:.4f}")
    np.savez(config["pinn_pred_path"], u_pred=u_pred.cpu().numpy(), u_true=u_test.cpu().numpy(), X=X_test.cpu().numpy())
    with open(config["loss_path"], "wb") as f:
        pickle.dump(solver.loss_history, f)


def train_gnn():
    print("[INFO] Training GNN (Optimized)...")

    device = torch.device(config["device"])
    data_path = config["gnn_dataset_path"]

    # Load or generate dataset
    if os.path.exists(data_path):
        print(f"[INFO] Loading dataset from: {data_path}")
        dataset, coords = torch.load(data_path, weights_only=False)
    else:
        dataset, coords = build_static_graph(
            config["data_path"],
            config["shape_file_path"]
        )
        # torch.save((dataset, coords), data_path)

    dataset = [d for d in dataset if isinstance(d, Data) and 'x' in d and 'y' in d and 'edge_index' in d]

    # ========= Split and DataLoader =========
    split_idx = int(len(dataset) * config["train_test_split"])
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    pin_memory = config["device"].startswith("cuda")

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, pin_memory=pin_memory)

    # === Model and optimization setup ===
    model = GNNPredictor(
        in_dim=config["gnn_in_dim"],
        hidden_dim=config["gnn_hidden_dim"],
        out_dim=config["gnn_out_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.1,
        anneal_strategy='cos'
    )
    loss_fn = torch.nn.MSELoss()

    # === Training loop with early stopping ===
    best_val_loss = float('inf')
    patience = config.get("patience", 50)
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index).squeeze()
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % config["print_interval"] == 0 or epoch == 1:
            print(f"Epoch {epoch}/{config['epochs']} - Loss: {avg_loss:.4f}")

        # Early stopping logic
        if avg_loss < best_val_loss - 1e-4:
            best_val_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config["gnn_model_path"])
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    print(f"[INFO] Best model saved to {config['gnn_model_path']}")

    # === Evaluation ===
    model.load_state_dict(torch.load(config["gnn_model_path"], map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch.x, batch.edge_index).squeeze()
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # === Inverse transform ===
    scaler_y_path = config["gnn_scaler_y"]
    if os.path.exists(scaler_y_path):
        scaler_y = torch.load(scaler_y_path, weights_only=False)
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()
    else:
        print("[WARNING] Target scaler not found, skipping inverse transform.")

    if config.get("plot", True):
        plot_prediction_vs_truth(preds, targets)
    metrics = compute_metrics(preds, targets, apply_expm1=False)
    print(f"GNN RMSE (original scale): {metrics['rmse']:.4f}, MAE (original scale): {metrics['mae']:.4f}")
    np.savez(config["gnn_pred_path"], preds=preds, targets=targets, coords=coords)
    print("[INFO] Predictions saved to ", config["gnn_pred_path"] )


def main():
    parser = argparse.ArgumentParser(description="Train PINN or GNN model")
    parser.add_argument('--model', type=str, default="pinn", choices=["pinn", "gnn"],
                        help="Choose model type to train: pinn or gnn")
    args = parser.parse_args()

    if args.model == "pinn":
        train_pinn()
    elif args.model == "gnn":
        train_gnn()
    else:
        raise ValueError("Invalid model choice. Use 'pinn' or 'gnn'.")


if __name__ == "__main__":
    main()
