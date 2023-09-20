import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from single_species_utils import init_lattice, update_wellmixed, update, update_stochastic

def ode_integrate(L, s, d, r, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the single species model.

    Parameters
    ----------
    L : int
        Side length of the square lattice.
    s : float
        Soil filling rate.
    d : float
        Death rate.
    r : float
        Reproduction rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    
    Returns
    -------
    T : list
        List of times.
    S : list
        List of soil fractions.
    E : list
        List of empty fractions.
    B : list
        List of bacteria fractions.
    """

    N_sites = L**2  # number of sites
    B_0 = int(N_sites / 10)  # initial number of bacteria
    E_0 = int((N_sites - B_0) / 2)  # initial number of empty sites
    S_0 = N_sites - B_0 - E_0  # initial number of soil sites

    dt = stoptime / nsteps

    S = [S_0/N_sites]
    B = [B_0/N_sites]
    E = [E_0/N_sites]
    T = [0]


    for i in tqdm(range(nsteps)):
        S.append(S[i] + dt * (s*E[i] - B[i]*S[i]))
        E.append(E[i] + dt * (B[i]*S[i] + d*B[i] - s*E[i] - r*B[i]*S[i]*E[i]))
        B.append(B[i] + dt * (r*B[i]*S[i]*E[i] - d*B[i]))
        T.append(T[i] + dt)
    
    return T, S, E, B


def main():

    # initialize the parameters
    n_steps = 10_000  # number of bacteria moves
    L = 20  # side length of the square lattice
    N = int(L**2 / 10)  # initial number of bacteria
    r = 1  # reproduction rate
    d = 0.1  # death rate
    s = 0.1  # soil filling rate
    soil_lattice = init_lattice(L, N)

    n_frames = 100  # number of potential frames in the animation (will be less in practice because only unique frames are saved)
    datasteps = np.geomspace(1, n_steps, n_frames, dtype=np.int32)  # steps at which to save the data
    datasteps = np.unique(datasteps)  # remove duplicates
    n_frames = len(datasteps)  # update the number of frames

    soil_lattice_data = np.zeros((len(datasteps), L, L), dtype=np.int8)
    soil_lattice_data[0] = soil_lattice

    # run the simulation
    for step in tqdm(range(1, n_steps+1)):
        update_wellmixed(soil_lattice, L, r, d, s)
        if step in datasteps:
            soil_lattice_data[np.where(datasteps == step)] = soil_lattice


    soil_lattice = init_lattice(L, N)
    soil_lattice_data_normal = np.zeros((len(datasteps), L, L), dtype=np.int8)
    soil_lattice_data_normal[0] = soil_lattice

    for step in tqdm(range(1, n_steps+1)):
        update(soil_lattice, L, r, d, s)
        if step in datasteps:
            soil_lattice_data_normal[np.where(datasteps == step)] = soil_lattice


    # run the ODE integrator
    T, S, E, B = ode_integrate(L, s, d, r, stoptime=n_steps, nsteps=100_000)

    # # animate the lattice
    # fig, ax = plt.subplots()
    # ax.set_xticks(np.arange(-.5, L, 1), minor=True)
    # ax.set_yticks(np.arange(-.5, L, 1), minor=True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(which='minor', linewidth=1)
    # ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[0]}")
    # im = ax.imshow(soil_lattice_data[0], cmap="cubehelix_r", vmin=0, vmax=2)
    # def animate(i):
    #     ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[i]}")
    #     im.set_data(soil_lattice_data[i])
    #     return im,
    # ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000, blit=True)
    # ani.save("src/single_species_RW/wellmixed_logspace.gif", fps=1)

    # plot the number of bacteria, soil, and empty sites as a function of time
    n_bacteria = np.zeros(n_frames, dtype=np.int32)
    n_soil = np.zeros(n_frames, dtype=np.int32)
    n_empty = np.zeros(n_frames, dtype=np.int32)
    for i, step in enumerate(datasteps):
        n_bacteria[i] = np.count_nonzero(soil_lattice_data[i] == 2)
        n_soil[i] = np.count_nonzero(soil_lattice_data[i] == 1)
        n_empty[i] = np.count_nonzero(soil_lattice_data[i] == 0)

    n_bacteria_normal = np.zeros(n_frames, dtype=np.int32)
    n_soil_normal = np.zeros(n_frames, dtype=np.int32)
    n_empty_normal = np.zeros(n_frames, dtype=np.int32)
    for i, step in enumerate(datasteps):
        n_bacteria_normal[i] = np.count_nonzero(soil_lattice_data_normal[i] == 2)
        n_soil_normal[i] = np.count_nonzero(soil_lattice_data_normal[i] == 1)
        n_empty_normal[i] = np.count_nonzero(soil_lattice_data_normal[i] == 0)

    fig, ax = plt.subplots()
    ax.plot(datasteps, n_bacteria, label="bacteria", c="forestgreen")
    # ax.plot(datasteps, n_soil, label="soil", c="darkgoldenrod")
    # ax.plot(datasteps, n_empty, label="empty", c="grey")
    ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}")
    # plot the ODE results
    ax.plot(T, np.array(B)*L**2, label="bacteria (ODE)", c="forestgreen", ls="--")
    # ax.plot(T, np.array(S)*L**2, label="soil (ODE)", c="darkgoldenrod", ls="--")
    # ax.plot(T, np.array(E)*L**2, label="empty (ODE)", c="grey", ls="--")
    ax.plot(datasteps, n_bacteria_normal, label="bacteria_neighbours", ls="-.", c="forestgreen")
    # # plot the totals
    # ax.plot(datasteps, n_bacteria + n_soil + n_empty, label="total", c="black")
    # ax.plot(T, np.array(B)*L**2 + np.array(S)*L**2 + np.array(E)*L**2, label="total (ODE)", c="black", ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("number of sites")
    ax.legend()
    plt.show()
    # save figure
    # fig.savefig("src/single_species_RW/wellmixed_time_ev.png", dpi=300)

    

if __name__ == "__main__":
    main()
