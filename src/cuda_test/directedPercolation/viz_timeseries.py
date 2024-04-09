import numpy as np
import matplotlib.pyplot as plt
import glob
import re

def main():
    # Get all .csv files in the directory
    files = glob.glob('src/cuda_test/directedPercolation/outputs/timeseries2D/*.csv')

    for file in files:
        # Use regex to get the p value from the filename
        match = re.search(r'p_(\d+\.?\d*)\.csv', file)
        if match:
            p = match.group(1)

            occupiedfracs = np.loadtxt(file)
            plt.plot(occupiedfracs, label=f'p={p}')

    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title('Occupied fraction vs time')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
