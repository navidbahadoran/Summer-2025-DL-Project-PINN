import torch
import torch.nn as nn
from config import config


class PINNSolver:
    def __init__(self, model, X_u_train, u_train, X_f_train,
                 X_ic=None, u_ic=None, X_bc=None, u_bc=None, device="cpu",
                 lambda_ic=1.0, lambda_bc=1.0):
        self.model = model.to(device)
        self.X_u = X_u_train.to(device)
        self.u = u_train.to(device)
        self.X_f = X_f_train.to(device)
        self.X_ic = X_ic.to(device) if X_ic is not None else None
        self.u_ic = u_ic.to(device) if u_ic is not None else None
        self.X_bc = X_bc.to(device) if X_bc is not None else None
        self.u_bc = u_bc.to(device) if u_bc is not None else None
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
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

        beta = 1.0  # fixed growth rate
        f = u_t - (u_xx + u_yy) - beta * u_pred

        return f

    def train(self, epochs=config["epochs"], lr=None, use_lbfgs=True):
        lr = lr or config["learning_rate"]
        adam = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()

        def closure():
            adam.zero_grad()
            u_pred = self.model(self.X_u)
            mse_u = mse_loss(u_pred, self.u)
            f_pred = self.net_residual(self.X_f)
            mse_f = mse_loss(f_pred, torch.zeros_like(f_pred).to(self.device))
            loss = mse_u + mse_f

            if self.X_ic is not None and self.u_ic is not None:
                ic_pred = self.model(self.X_ic)
                loss += self.lambda_ic * mse_loss(ic_pred, self.u_ic)

            if self.X_bc is not None and self.u_bc is not None:
                bc_pred = self.model(self.X_bc)
                loss += self.lambda_bc * mse_loss(bc_pred, self.u_bc)

            loss.backward()
            return loss

        # Warmup with Adam
        print("[INFO] Starting Adam warm-up phase")
        for epoch in range(epochs):
            self.model.train()
            adam.zero_grad()

            u_pred = self.model(self.X_u)
            mse_u = mse_loss(u_pred, self.u)
            f_pred = self.net_residual(self.X_f)
            mse_f = mse_loss(f_pred, torch.zeros_like(f_pred).to(self.device))

            mse_ic = torch.tensor(0.0, device=self.device)
            mse_bc = torch.tensor(0.0, device=self.device)

            if self.X_ic is not None and self.u_ic is not None:
                mse_ic = mse_loss(self.model(self.X_ic), self.u_ic)

            if self.X_bc is not None and self.u_bc is not None:
                mse_bc = mse_loss(self.model(self.X_bc), self.u_bc)

            loss = mse_u + mse_f + self.lambda_ic * mse_ic + self.lambda_bc * mse_bc
            loss.backward()
            adam.step()
            self.loss_history.append(loss.item())

            if epoch % config["print_interval"] == 0 or epoch == 999:
                print(f"[Adam {epoch:05d}] Loss: {loss.item():.6e}")

        # LBFGS Phase
        if use_lbfgs:
            print("[INFO] Starting LBFGS phase")
            lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                      max_iter=500,
                                      history_size=50,
                                      tolerance_grad=1e-7,
                                      tolerance_change=1e-9,
                                      line_search_fn='strong_wolfe')
            lbfgs.step(closure)
            print("[INFO] LBFGS optimization completed.")
