import torch
import torch.nn as nn
from config import config


class PINNSolver:


    def __init__(self, model, X_u_train, u_train, X_f_train, device="cpu"):
        self.model = model.to(device)
        self.X_u = X_u_train.to(device)
        self.u = u_train.to(device)
        self.X_f = X_f_train.to(device)
        self.device = device
        self.loss_history = []

    def net_residual(self, X_f):
        X_f = X_f.clone().detach().requires_grad_(True).to(self.device)
        u_pred = self.model(X_f)

        grads = torch.autograd.grad(
            outputs=u_pred,
            inputs=X_f,
            grad_outputs=torch.ones_like(u_pred).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0]

        u_t = grads[:, 2:3]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, X_f,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            u_y, X_f,
            grad_outputs=torch.ones_like(u_y).to(self.device),
            create_graph=True,
            retain_graph=True
        )[0][:, 1:2]

        # PDE residual: ∂u/∂t = ∂²u/∂x² + ∂²u/∂y²
        beta = 1.0  # a learnable or fixed growth rate
        f = u_t - (u_xx + u_yy) - beta * u_pred

        return f

    def train(self, epochs=config["epochs"], lr=None):
        lr = lr or config["learning_rate"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            u_pred = self.model(self.X_u)
            mse_u = mse_loss(u_pred, self.u)

            f_pred = self.net_residual(self.X_f)
            mse_f = mse_loss(f_pred, torch.zeros_like(f_pred).to(self.device))

            loss = mse_u + mse_f
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())

            if epoch % 500 == 0 or epoch == epochs - 1:
                print(f"[{epoch:05d}] Total Loss: {loss.item():.6e} | Supervised: {mse_u.item():.6e} | Residual: {mse_f.item():.6e}")