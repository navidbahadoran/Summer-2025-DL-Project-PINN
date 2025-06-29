import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation_fd(nx=50, ny=50, nt=100, T=1.0, alpha=1.0):
    """
    Solves the 2D heat equation using finite differences.

    Returns:
        u_fd: (nt, ny, nx) array of u(x, y, t)
        X, Y: meshgrid for spatial domain
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = 0.25 * min(dx, dy)**2  # conservative CFL-like setting


    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial condition: square hot spot
    u = np.zeros((nx, ny))
    u[int(nx * 0.4):int(nx * 0.6), int(ny * 0.4):int(ny * 0.6)] = 1.0

    # Store full time evolution
    u_fd = np.zeros((nt, nx, ny), dtype=np.float32)
    u_fd[0] = u.copy()

    for n in range(1, nt):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + alpha * dt / dx**2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            + alpha * dt / dy**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        u = u_new
        u_fd[n] = u.copy()

    return u_fd, X, Y


def plot_fd_solution(u, X, Y, title="Finite Difference Solution"):
    """
    Plots the final time step as a 3D surface.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y, T)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def save_fd_solution(u_fd, X, Y, filepath="fd_solution.npz"):
    """
    Saves the full time-evolved FD solution to compressed .npz
    """
    np.savez_compressed(filepath, u=u_fd, X=X, Y=Y)
    print(f"FD solution saved to {filepath}")


if __name__ == "__main__":
    u_fd, X, Y = solve_heat_equation_fd(nx=100, ny=100, nt=100, T=1.0)
    plot_fd_solution(u_fd[-1], X, Y)  # Plot last time step
    save_fd_solution(u_fd, X, Y)
