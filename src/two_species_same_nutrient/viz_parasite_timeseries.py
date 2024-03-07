import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main():
    # 3D
    L = 50
    # sigma = 1
    # theta = 0.03
    sigma = 0.5
    theta = 0.0395

    step, E, N, S, G, B = np.loadtxt(f"src/nutrient_mutations/outputs/timeseries/sigma_{sigma}_theta_{theta}.csv", skiprows=1, delimiter=",", unpack=True)

    fig, axs = plt.subplots(figsize=(10, 6))

    axs.plot(step, S, label="soil", c="brown")
    axs.plot(step, E, label="emptys", c="grey")
    axs.plot(step, N, label="nutrients", c="lawngreen")
    axs.plot(step, G, label="green worms", c="green")
    axs.plot(step, B, label="blue worms", c="blue")
    axs.set_title(f"{L=}, {sigma=}, {theta=}")
    axs.set_xlabel(r"Timestep / L$^3$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    plt.savefig('src/two_species_same_nutrient/plots/lattice_timeseries/parasite_oscillations.png', dpi=300)

    plt.show()


def plot_from_cuda_bin():
    L = 1024
    sigma = 1
    theta = 0.042
    rhofactor = 4
    n_steps = 10000
    portion_size = L * L
    record_every = 10  # Record every __ steps

    filename = f'docs/data/twospec_samenutrient/lattice_anim_L_{L}_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.bin'

    # Initialize lists to store counts
    E, N, S, G, B = [], [], [], [], []

    with open(filename, 'rb') as f:
        for current_chunk in tqdm(range(n_steps)):
            if current_chunk % record_every == 0:  # Only calculate timestep every record_every steps
                f.seek(current_chunk * portion_size)
                portion = np.fromfile(f, dtype=np.uint8, count=portion_size)

                # Count occurrences of each state
                counts = np.bincount(portion, minlength=5)
                E.append(counts[0])
                N.append(counts[1])
                S.append(counts[2])
                G.append(counts[3])
                B.append(counts[4])

    # Convert lists to numpy arrays for plotting
    E, N, S, G, B = map(np.array, [E, N, S, G, B])
    step = np.arange(0, n_steps, record_every)

    fig, axs = plt.subplots(figsize=(10, 6))

    axs.plot(step, S, label="soil", c="brown")
    axs.plot(step, E, label="emptys", c="grey")
    axs.plot(step, N, label="nutrients", c="lawngreen")
    axs.plot(step, G, label="green worms", c="green")
    axs.plot(step, B, label="blue worms", c="blue")
    axs.set_title(f"{L=}, {sigma=}, {theta=}")
    axs.set_xlabel(r"Timestep / L$^3$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    plt.savefig(f'src/two_species_same_nutrient/plots/lattice_timeseries/confinementtest/2D_{sigma=}_{theta=}_{rhofactor=}.png', dpi=300)

    plt.show()


if __name__ == "__main__":
    # main()
    plot_from_cuda_bin()
