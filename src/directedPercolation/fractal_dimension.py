import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def box_count(lattice, sizes):
    counts = []
    for size in sizes:
        count = 0
        for i in range(0, lattice.shape[0], size):
            for j in range(0, lattice.shape[1], size):
                if np.any(lattice[i:i+size, j:j+size]):
                    count += 1
        counts.append(count)
    return counts

def fractal_dimension(lattice, max_box_size):
    # sizes = [2**i for i in range(int(np.log2(max_box_size))+1)]
    sizes = np.arange(2, max_box_size+1)
    counts = box_count(lattice, sizes)
    coeffs, _ = curve_fit(lambda x, a, b: a + b*x, np.log(sizes), np.log(counts))
    return coeffs, sizes, counts

def main():
    p = 0.2873
    L = 2048
    steps = 2000

    lattice = np.loadtxt(f'src/directedPercolation/outputs/lattice2D/survival_p_{p}_L_{L}_steps_{steps}.csv')

    plt.imshow(lattice, cmap='binary')
    plt.show()

    max_box_size = L // 64
    coeffs, sizes, counts = fractal_dimension(lattice, max_box_size)
    print(f'Fractal dimension: {-coeffs[1]}')

    # Plotting
    plt.loglog(sizes, counts, 'x', label='Data')
    plt.loglog(sizes, [np.exp(coeffs[0]) * size**coeffs[1] for size in sizes], '-', label=f'Fit, D = {-coeffs[1]:.2f}')
    plt.xlabel('Box size')
    plt.ylabel('Number of boxes')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()