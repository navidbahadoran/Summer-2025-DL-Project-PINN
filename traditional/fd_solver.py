import numpy as np

def fd_solver_1d_reaction_diffusion(
    L=1.0, T=1.0, Nx=100, Nt=1000, D=0.01, r=1.0, K=1.0,
    initial_condition=None, boundary_condition=None
):
    dx = L / (Nx - 1)
    dt = T / Nt
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt+1)

    # Stability condition (optional warning)
    if D * dt / dx**2 > 0.5:
        print(f"[WARNING] Unstable scheme: D*dt/dx^2 = {D * dt / dx**2:.2f} > 0.5")

    # Initialize solution matrix
    u = np.zeros((Nt+1, Nx))
    
    # Set initial condition
    u[0, :] = initial_condition(x)

    # Time stepping loop
    for n in range(0, Nt):
        for i in range(1, Nx - 1):
            diffusion = D * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            reaction = r * u[n, i] * (1 - u[n, i] / K)
            u[n+1, i] = u[n, i] + dt * (diffusion + reaction)

        # Apply boundary condition (Dirichlet)
        u[n+1, 0] = boundary_condition(t[n+1])
        u[n+1, -1] = boundary_condition(t[n+1])

    return x, t, u
