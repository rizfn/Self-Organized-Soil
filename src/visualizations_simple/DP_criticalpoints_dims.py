import numpy as np
import matplotlib.pyplot as plt 
from cycler import cycler

def main():
    dims = np.arange(1, 6)
    DP_points = np.array([0.6447, 0.2873, 0.1324, 0.0638, 0.0315])
    espl_points = np.array([0.6447, 0.318, 0.201, 0.1665, 0.11])
    fspl_points = np.array([1, 0.343, 0.146, 0.065, 0.0317])

    color_cycle = cycler(color=['#666666', '#901A1E', '#17BEBB'])
    plt.rcParams['axes.prop_cycle'] = color_cycle

    plt.plot(DP_points, dims, label='DP', marker='o', linestyle='--')
    plt.plot(espl_points, dims, label='Empty power-law', marker='o', linestyle='--')
    plt.plot(fspl_points, dims, label='Filled power-law', marker='o', linestyle='--')
    # plt.plot(np.exp(-dims), dims, label='exponential', marker='.', linestyle='--')
    # plt.plot(1/dims, dims, label='powerlaw', marker='.', linestyle='--')
    plt.ylabel('Dimensions')
    plt.xlabel('Critical $p$')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('$d$ vs $p$, log-log scale')
    plt.title('$d$ vs $p$')
    # plt.savefig('src/visualizations_simple/plots/DP_criticalpoints_dims_linear.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    plt.plot(DP_points * 2**dims, dims, label='DP', marker='o', linestyle='--')
    plt.plot(espl_points * 2**dims, dims, label='Empty power-law', marker='o', linestyle='--')
    plt.plot(fspl_points * 2**dims, dims, label='Filled power-law', marker='o', linestyle='--')
    plt.ylabel('Dimensions')
    plt.xlabel('Critical $p$, normalized by number of neighbours ($2^d$)')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.title('$d$ vs $p \\times 2^d$')
    # plt.savefig('src/visualizations_simple/plots/DP_criticalpoints_dims_normalized.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()



if __name__ == "__main__":
    main()