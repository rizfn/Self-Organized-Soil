import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numba import njit


@njit
def ode_derivatives(S, E, N, W, sigma, theta, rho, delta):
    """Calculate the derivatives of S, E, N, W.

    This function is not called directly, but rather through `ode_integrate_rk4`
    
    Parameters
    ----------
    S : float
        Soil fraction.
    E : float
        Empty fraction.
    N : float
        Nutrient fraction.
    W : float
        Worm fraction.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm reproduction rate.
    delta : float
        Nutrient decay rate.

    Returns
    -------
    dS : float
        Derivative of soil fraction.
    dE : float
        Derivative of empty fraction.
    dN : float
        Derivative of nutrient fraction.
    dW : float
        Derivative of worm fraction.
    """

    dS = sigma*S*(E+N) - W*S
    dE = (1-rho)*W*N + theta*W - sigma*S*E + delta*N
    dN = W*S - W*N - sigma*S*N - delta*N
    dW = rho*W*N - theta*W

    return dS, dE, dN, dW


@njit
def ode_integrate_rk4(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000, S_0=0.3, E_0=0.3, N_0=0.3, W_0=0.1):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method.
    
    Parameters
    ----------
    
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm reproduction rate.
    delta : float
        Nutrient decay rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100_000.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    """

    dt = stoptime / nsteps

    S = np.zeros(nsteps+1)
    W = np.zeros(nsteps+1)
    E = np.zeros(nsteps+1)
    N = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    S[0] = S_0
    W[0] = W_0
    E[0] = E_0
    N[0] = N_0
    T[0] = 0

    for i in range(nsteps):
        k1_S, k1_E, k1_N, k1_W = ode_derivatives(S[i], E[i], N[i], W[i], sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k1_S
        E_temp = E[i] + 0.5 * dt * k1_E
        N_temp = N[i] + 0.5 * dt * k1_N
        W_temp = W[i] + 0.5 * dt * k1_W
        k2_S, k2_E, k2_N, k2_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k2_S
        E_temp = E[i] + 0.5 * dt * k2_E
        N_temp = N[i] + 0.5 * dt * k2_N
        W_temp = W[i] + 0.5 * dt * k2_W
        k3_S, k3_E, k3_N, k3_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + dt * k3_S
        E_temp = E[i] + dt * k3_E
        N_temp = N[i] + dt * k3_N
        W_temp = W[i] + dt * k3_W
        k4_S, k4_E, k4_N, k4_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S[i+1] = S[i] + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E[i+1] = E[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        N[i+1] = N[i] + (dt / 6) * (k1_N + 2 * k2_N + 2 * k3_N + k4_N)
        W[i+1] = W[i] + (dt / 6) * (k1_W + 2 * k2_W + 2 * k3_W + k4_W)
        T[i+1] = T[i] + dt

    return T, S, E, N, W


def stable_fixed_point(sigma, theta):
    sqrt_term = np.sqrt(sigma**2 * theta**2 - 2 * sigma * theta - 4 * theta + 1)
    E = (sqrt_term - sigma * theta - 2 * theta + 1) / (2 * (sigma + 1))
    S = 0.5 * (-sqrt_term - sigma * theta + 1)
    W = sigma * (sqrt_term + sigma * theta + 1) / (2 * (sigma + 1))
    return E, S, W

def unstable_fixed_point(sigma, theta):
    sqrt_term = np.sqrt(sigma**2 * theta**2 - 2 * sigma * theta - 4 * theta + 1)
    E = (-sqrt_term - sigma * theta - 2 * theta + 1) / (2 * (sigma + 1))
    S = 0.5 * (sqrt_term - sigma * theta + 1)
    W = sigma * (-sqrt_term + sigma * theta + 1) / (2 * (sigma + 1))
    return E, S, W

def nullcline_intersections(sigma, theta):
    # S, E nullcline intersection
    def SE_intersection(E):
        W = E * sigma / (E * (sigma + 1) + theta)
        S = theta / (E * (sigma + 1) + theta)
        return S, E, W

    # S, W nullcline intersection
    def SW_intersection(E):
        W = sigma * E + sigma * theta
        S = 1 - E * (1 + sigma) - theta * (1 + sigma)
        return S, E, W

    # E, W nullcline intersection
    def EW_intersection(E):
        W = (sigma * E - sigma * E**2 - sigma * E * theta) / (theta + sigma * E)
        S = 1 - E - W - theta
        return S, E, W

    E_values = np.linspace(0.0001, 1, 100)  # Avoid division by zero
    SE_points = [SE_intersection(E) for E in E_values]
    SW_points = [SW_intersection(E) for E in E_values]
    EW_points = [EW_intersection(E) for E in E_values]

    SE_points = [point for point in SE_points if all(0 <= coord <= 1 for coord in point)]
    SW_points = [point for point in SW_points if all(0 <= coord <= 1 for coord in point)]
    EW_points = [point for point in EW_points if all(0 <= coord <= 1 for coord in point)]


    return SE_points, SW_points, EW_points


def main():
    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta = 0.1
    sigma = 0.39

    E0, S0, W0 = stable_fixed_point(sigma, theta)
    E0 = E0 - 0.001  # slightly perturb the stable fixed point
    N0 = 1 - E0 - S0 - W0

    T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps, S_0=S0, E_0=E0, N_0=N0, W_0=W0)

    # Plot the trajectory in E, S, W space
    fig = plt.figure(figsize=(20, 8))

    SE_points, SW_points, EW_points = nullcline_intersections(sigma, theta)

    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot nullcline intersections
    SE_points = np.array(SE_points)
    SW_points = np.array(SW_points)
    EW_points = np.array(EW_points)
    ax1.plot(SE_points[:, 1], SE_points[:, 0], SE_points[:, 2], color='cyan', alpha=0.5, label='S,E nullcline intersection')
    ax1.plot(SW_points[:, 1], SW_points[:, 0], SW_points[:, 2], color='magenta', alpha=0.5, label='S,W nullcline intersection')
    ax1.plot(EW_points[:, 1], EW_points[:, 0], EW_points[:, 2], color='yellow', alpha=0.5, label='E,W nullcline intersection')

    ax1.plot(E, S, W, label='Trajectory', color='b')
    ax1.scatter(*stable_fixed_point(sigma, theta), color='r', label='Stable fixed point')
    ax1.scatter(*unstable_fixed_point(sigma, theta), color='g', label='Unstable fixed point')

    ax1.set_xlabel('E')
    ax1.set_ylabel('S')
    ax1.set_zlabel('W')
    ax1.set_title('State Space Trajectory in E, S, W Space')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.legend()

    # Plot E, S, W, N over time
    ax2 = fig.add_subplot(122)
    ax2.plot(T, N, label='N', color='limegreen')
    ax2.plot(T, E, label='E', color='grey')
    ax2.plot(T, W, label='W', color='green')
    ax2.plot(T, S, label='S', color='maroon')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fraction')
    ax2.set_title('E, S, W, N over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()