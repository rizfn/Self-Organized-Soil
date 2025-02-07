import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define the nullcline equations
def nullcline_S(W, E, sigma, theta):
    return (sigma - W * (sigma + 1)) / sigma

def nullcline_E(W, E, sigma, theta):
    return (theta * W) / (sigma * E)

def nullcline_W(W, E, sigma, theta):
    return 1 - E - W - theta

# Parameters
theta = 0.1
sigma_values = [0.1, 0.39, 0.4, 1]

# Create a grid for E and W
E = np.linspace(0.01, 1, 100)  # Avoid division by zero
W = np.linspace(0, 1, 100)
E_grid, W_grid = np.meshgrid(E, W)

# Function to calculate the fixed points
def unstable_fixed_point(sigma, theta):
    sqrt_term = np.sqrt(sigma**2 * theta**2 - 2 * sigma * theta - 4 * theta + 1)
    E = (-sqrt_term - sigma * theta - 2 * theta + 1) / (2 * (sigma + 1))
    S = 0.5 * (sqrt_term - sigma * theta + 1)
    W = sigma * (-sqrt_term + sigma * theta + 1) / (2 * (sigma + 1))
    return E, S, W

def stable_fixed_point(sigma, theta):
    sqrt_term = np.sqrt(sigma**2 * theta**2 - 2 * sigma * theta - 4 * theta + 1)
    E = (sqrt_term - sigma * theta - 2 * theta + 1) / (2 * (sigma + 1))
    S = 0.5 * (-sqrt_term - sigma * theta + 1)
    W = sigma * (sqrt_term + sigma * theta + 1) / (2 * (sigma + 1))
    return E, S, W

# Plot the nullclines
fig = plt.figure(figsize=(16, 12))

for i, sigma in enumerate(sigma_values):
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    
    # Calculate the nullclines
    NullclineS = nullcline_S(W_grid, E_grid, sigma, theta)
    NullclineE = nullcline_E(W_grid, E_grid, sigma, theta)
    NullclineW = nullcline_W(W_grid, E_grid, sigma, theta)
    
    # Plot the surfaces for nullclines
    ax.plot_surface(E_grid, NullclineS, W_grid, color='r', alpha=0.4, label='Nullcline S')
    ax.plot_surface(E_grid, NullclineE, W_grid, color='g', alpha=0.4, label='Nullcline E')
    ax.plot_surface(E_grid, NullclineW, W_grid, color='b', alpha=0.4, label='Nullcline W')

    # Calculate and plot the fixed points
    E_unstable, S_unstable, W_unstable = unstable_fixed_point(sigma, theta)
    E_stable, S_stable, W_stable = stable_fixed_point(sigma, theta)
    
    ax.scatter(E_unstable, S_unstable, W_unstable, color='k', s=100, label='Unstable fixed point')
    ax.scatter(E_stable, S_stable, W_stable, color='m', s=100, label='Stable fixed point')

    ax.set_xlabel('E')
    ax.set_ylabel('S')
    ax.set_zlabel('W')
    ax.set_title(f'Nullclines for sigma = {sigma}')

    # Set the axis limits to 0, 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Create custom legend handles
    custom_lines = [plt.Line2D([0], [0], color='r', lw=4),
                    plt.Line2D([0], [0], color='g', lw=4),
                    plt.Line2D([0], [0], color='b', lw=4),
                    plt.Line2D([0], [0], color='k', marker='o', markersize=10, linestyle='None'),
                    plt.Line2D([0], [0], color='m', marker='o', markersize=10, linestyle='None')]
    ax.legend(custom_lines, ['Nullcline S', 'Nullcline E', 'Nullcline W', 'Unstable fixed point', 'Stable fixed point'])

plt.tight_layout()
plt.show()