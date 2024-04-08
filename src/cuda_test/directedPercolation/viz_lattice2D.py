import numpy as np
import matplotlib.pyplot as plt


def main():
    p = 0.3
    L = 100
    CUDA = 1

    filename = f'src/cuda_test/directedPercolation/outputs/lattice2D/{"CUDA_" if CUDA else ""}p_{p}_L_{L}.csv'
    print(f'Loading {filename}')
    # Load the CSV file
    lattice = np.loadtxt(filename, delimiter=',')

    # Display the final lattice using imshow
    plt.imshow(lattice, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
