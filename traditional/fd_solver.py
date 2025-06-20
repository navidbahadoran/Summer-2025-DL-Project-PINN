# traditional/fd_solver.py

import numpy as np
import matplotlib.pyplot as plt


def solve_heat_equation_fd(nx=50, ny=50, nt=100, dt=0.001, alpha=1.0):
    """
    Solve the 2D heat equation:
        ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    using explicit finite difference method.

    Parameters:
    - nx, ny: number of grid points in x and y
    - nt: number of time steps
    - dt: time step size
    - alpha: thermal diffusivity (default=1.0)

    Returns:
    - u: solution at final time step
    - x, y: spatial grid coordinates
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Initial condition: hot square in the center
    u = np.zeros((ny, nx))
    u[int(ny*0.4):int(ny*0.6), int(nx*0.4):int(nx*0.6)] = 1.0

    for n in range(nt):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + alpha * dt / dx**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
            + alpha * dt / dy**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        )
        u = u_new

    return u, X, Y

def plot_fd_solution(u, X, Y, title="Finite Difference Solution"):
    """
    Plot the final solution surface from FD.
    """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y, T)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    u, X, Y = solve_heat_equation_fd()
    plot_fd_solution(u, X, Y)
