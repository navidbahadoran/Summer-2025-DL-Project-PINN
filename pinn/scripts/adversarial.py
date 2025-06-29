import torch
import numpy as np

def perturb_initial_conditions(X_ic, u_ic, epsilon=0.05):
    noise = epsilon * torch.randn_like(u_ic)
    return X_ic.clone(), u_ic + noise

def perturb_boundary_conditions(X_bc, u_bc, epsilon=0.05):
    noise = epsilon * torch.randn_like(u_bc)
    return X_bc.clone(), u_bc + noise

def perturb_collocation_points(X_f, scale=0.01):
    delta = scale * (2 * torch.rand_like(X_f) - 1.0)
    return torch.clamp(X_f + delta, 0.0, 1.0)  # keep in domain

# Example usage:
if __name__ == "__main__":
    X_ic = torch.rand(100, 3)
    u_ic = torch.sin(X_ic[:, 0:1])
    X_bc = torch.rand(100, 3)
    u_bc = torch.cos(X_bc[:, 1:2])
    X_f = torch.rand(10000, 3)

    X_ic_p, u_ic_p = perturb_initial_conditions(X_ic, u_ic)
    X_bc_p, u_bc_p = perturb_boundary_conditions(X_bc, u_bc)
    X_f_p = perturb_collocation_points(X_f)

    print("Perturbed IC/BC/Collocation created.")
