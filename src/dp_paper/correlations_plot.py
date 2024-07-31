import numpy as np
import matplotlib.pyplot as plt
import glob


def main():
    # dir = 'src/dp_paper/outputs/corr2D/activePoint/'
    dir = 'src/dp_paper/outputs/corr2D/sameCluster/'
    files = glob.glob(dir + '*.tsv')

    for file in files:
        dist, corr = np.loadtxt(file, unpack=True, delimiter='\t')
        p = float(file.split('\\')[-1].split('_')[1].replace('p=', ''))
        L = int((file.split('\\')[-1].split('_')[-1]).split('.')[0])
        plt.plot(dist[1:], corr[1:], label=f'{p=}, {L=}', alpha=0.8)
        # plt.plot(dist[1:], corr[1:], label=file.split('\\')[-1], linestyle='-', alpha=0.8)


    x = dist[1:]
    nu = 2
    plt.plot(x, 2*corr[1]*x**(-nu), label=f'$\\nu$={nu} power law', linestyle='--', alpha=0.8)

    plt.grid()
    plt.legend()
    plt.title('Probability of neighbour at distance $x$ being in the same cluster')
    plt.xlabel('Manhattan distance')
    plt.ylabel('Correlation (observed/expected)')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()
