import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from two_species_utils import ode_integrate, run_stochastic, run_stochastic_wellmixed


def main():

    # initialize the parameters
    n_steps = 1_000_000  # number of bacteria moves
    L = 20  # side length of the square lattice
    N = int(L**2 / 10)  # initial number of bacteria
    r = 1  # reproduction rate
    d = 0.05  # death rate
    s = 0.1  # soil filling rate

    n_frames = 100  # number of potential frames in the animation (will be less in practice because only unique frames are saved)
    datasteps = np.geomspace(1, n_steps, n_frames, dtype=np.int32)  # steps at which to save the data
    datasteps = np.unique(datasteps)  # remove duplicates
    n_frames = len(datasteps)  # update the number of frames

    # run the simulation with well-mixed interactions
    soil_lattice_data_wellmixed = run_stochastic_wellmixed(n_steps, L, r, d, s, datasteps)

    # # run the simulation with nearest neighbour interactions
    # soil_lattice_data = run_stochastic(n_steps, L, r, d, s, datasteps)

    # run the ODE integrator
    T, S, E_R, E_B, R, B = ode_integrate(s, d, r, n_steps//L**2, n_steps//L**2)

    two_spec_cmap = LinearSegmentedColormap.from_list("two_spec", ['#fcb6b1', '#b1cafc', '#996953', '#cc3535','#2861c9'], N=5)

    # # animate the lattice
    # fig, ax = plt.subplots()
    # ax.set_xticks(np.arange(-.5, L, 1))
    # ax.set_yticks(np.arange(-.5, L, 1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(which='major', linewidth=1.5)
    # ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[0]}")
    # im = ax.imshow(soil_lattice_data_wellmixed[0], cmap=two_spec_cmap)
    # def animate(i):
    #     ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[i]}")
    #     im.set_data(soil_lattice_data_wellmixed[i])
    #     return im,
    # ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000, blit=True)
    # ani.save("src/two_species_RW/wellmixed_logspace.gif", fps=1)

    # plot the ODE results
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].plot(T, S, label="soil", c=two_spec_cmap(2))
    ax[0].plot(T, E_R, label="empty red", c=two_spec_cmap(0))
    ax[0].plot(T, E_B, label="empty blue", c=two_spec_cmap(1))
    ax[0].plot(T, R, label="red", c=two_spec_cmap(3))
    ax[0].plot(T, B, label="blue", c=two_spec_cmap(4))
    ax[0].set_title(f"{r=:.2f}, {d=:.2f}, {s=:.2f}")
    ax[0].legend()
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Number of sites")

    # plot the well-mixed results
    E_R_wellmixed = np.zeros(n_frames, dtype=np.int32)
    E_B_wellmixed = np.zeros(n_frames, dtype=np.int32)
    S_wellmixed = np.zeros(n_frames, dtype=np.int32)
    R_wellmixed = np.zeros(n_frames, dtype=np.int32)
    B_wellmixed = np.zeros(n_frames, dtype=np.int32)
    for i in range(n_frames):
        E_R_wellmixed[i] = np.count_nonzero(soil_lattice_data_wellmixed[i] == 0)
        E_B_wellmixed[i] = np.count_nonzero(soil_lattice_data_wellmixed[i] == 1)
        S_wellmixed[i] = np.count_nonzero(soil_lattice_data_wellmixed[i] == 2)
        R_wellmixed[i] = np.count_nonzero(soil_lattice_data_wellmixed[i] == 3)
        B_wellmixed[i] = np.count_nonzero(soil_lattice_data_wellmixed[i] == 4)
    E_R_wellmixed = E_R_wellmixed / L**2
    E_B_wellmixed = E_B_wellmixed / L**2
    S_wellmixed = S_wellmixed / L**2
    R_wellmixed = R_wellmixed / L**2
    B_wellmixed = B_wellmixed / L**2

    ax[1].plot(datasteps, S_wellmixed, label="soil", c=two_spec_cmap(2))
    ax[1].plot(datasteps, E_R_wellmixed, label="empty red", c=two_spec_cmap(0))
    ax[1].plot(datasteps, E_B_wellmixed, label="empty blue", c=two_spec_cmap(1))
    ax[1].plot(datasteps, R_wellmixed, label="red", c=two_spec_cmap(3))
    ax[1].plot(datasteps, B_wellmixed, label="blue", c=two_spec_cmap(4))
    ax[1].set_xscale("log")
    ax[1].set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Number of sites")
    plt.show()
    
    

if __name__ == "__main__":
    main()
