import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from single_species_utils import init_lattice, update

def main():

    # initialize the parameters
    n_steps = 100_000  # number of bacteria moves
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
        update(soil_lattice, L, r, d, s)
        if step in datasteps:
            soil_lattice_data[np.where(datasteps == step)] = soil_lattice

    # animate the lattice
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-.5, L, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='minor', linewidth=1)
    ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[0]}")
    im = ax.imshow(soil_lattice_data[0], cmap="cubehelix_r", vmin=0, vmax=2)
    def animate(i):
        ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}\nstep {datasteps[i]}")
        im.set_data(soil_lattice_data[i])
        return im,
    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000, blit=True)
    ani.save("src/single_species_RW/single_species_logspace.gif", fps=1)

    # plot the number of bacteria, soil, and empty sites as a function of time
    n_bacteria = np.zeros(n_frames, dtype=np.int32)
    n_soil = np.zeros(n_frames, dtype=np.int32)
    n_empty = np.zeros(n_frames, dtype=np.int32)
    for i, step in enumerate(datasteps):
        n_bacteria[i] = np.count_nonzero(soil_lattice_data[i] == 2)
        n_soil[i] = np.count_nonzero(soil_lattice_data[i] == 1)
        n_empty[i] = np.count_nonzero(soil_lattice_data[i] == 0)
    fig, ax = plt.subplots()
    ax.plot(datasteps, n_bacteria, label="bacteria")
    ax.plot(datasteps, n_soil, label="soil")
    ax.plot(datasteps, n_empty, label="empty")
    ax.set_title(f"{L=}, {r=:.2f}, {d=:.2f}, {s=:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("number of sites")
    ax.legend()
    # save figure
    fig.savefig("src/single_species_RW/single_species_time_ev.png", dpi=300)

    

if __name__ == "__main__":
    main()
