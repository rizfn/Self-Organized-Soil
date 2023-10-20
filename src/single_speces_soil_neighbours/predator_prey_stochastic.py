import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from soil_neighbour_utils import run_predatorprey


def main():

    # initialize the parameters
    # NOTE: 1M is fairly fast, but for some reason, going to 10M seriously slows down the code. Maybe memory issue?
    n_steps = 1_000_000  # number of bacteria moves
    L = 50  # side length of the square lattice
    r = 1  # reproduction rate
    d = 0.06  # death rate
    s = 0.4  # soil filling rate
    datasteps = np.arange(0, n_steps, 1000)

    soil_lattice_data = run_predatorprey(n_steps, L, r, d, s, datasteps)

    n_empty = np.zeros(len(datasteps), dtype=np.int16)
    n_soil = np.zeros(len(datasteps), dtype=np.int16)
    n_bacteria = np.zeros(len(datasteps), dtype=np.int16)

    for i in tqdm(range(len(soil_lattice_data))):
        n_empty[i] = np.sum(soil_lattice_data[i] == 0)
        n_soil[i] = np.sum(soil_lattice_data[i] == 1)
        n_bacteria[i] = np.sum(soil_lattice_data[i] == 2)

    plt.plot(datasteps, n_empty, label="empty")
    plt.plot(datasteps, n_soil, label="soil")
    plt.plot(datasteps, n_bacteria, label="bacteria")
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()
