import matplotlib.pyplot as plt
import numpy as np
import glob
        

def set_matplotlib_params():

    colors = ['#901A1E', '#16BAC5', '#666666']

    plt.rcParams.update({'font.size': 16, 
                        #  'font.family': 'MS Reference Sans Serif', 
                         'axes.grid': True, 
                         'axes.prop_cycle': plt.cycler('color', colors),  # Set the color cycle
                         'grid.linestyle': '-',
                         'grid.linewidth': 0.5, 
                         'grid.alpha': 0.8, })


def main():
    L = 1024
    files = glob.glob(f'src/dp_paper/outputs/orderParameter/L_{L}/*.tsv')

    p_list = []
    xspan_list = []
    yspan_list = []
    xspan_or_yspan_list = []

    for file in files:
        p = float(file.split('_')[-1].rsplit('.tsv')[0])
        step, xspan, yspan = np.loadtxt(file, unpack=True, delimiter='\t', skiprows=1)
        p_list.append(p)
        xspan_list.append(np.mean(xspan))
        yspan_list.append(np.mean(yspan))
        xspan_or_yspan_list.append(np.mean(np.logical_or(xspan, yspan)))

    # sort p_list and the other lists accordingly
    p_list, xspan_list, yspan_list, xspan_or_yspan_list = zip(*sorted(zip(p_list, xspan_list, yspan_list, xspan_or_yspan_list)))

    set_matplotlib_params()
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 9))
    axs[0].plot(p_list, xspan_list, marker='o', linestyle='--', label='xspan')
    axs[0].set_ylabel('$R^{(h)}$')
    axs[1].plot(p_list, yspan_list, marker='o', linestyle='--', label='yspan')
    axs[1].set_ylabel('$R^{(v)}$')
    axs[2].plot(p_list, xspan_or_yspan_list, marker='o', linestyle='--', label='xspan or yspan')
    axs[2].set_ylabel('$R^{(e)}$')
    axs[2].set_xlabel('p')

    plt.tight_layout()
    plt.savefig(f'src/dp_paper/plots/orderParameter/L_{L}.png', dpi=300)
    plt.show()

def systemSizeComparison():

    L_list = [512, 1024, 2048]
    xspan_vs_L_list = []

    for L in L_list:
        files = glob.glob(f'src/dp_paper/outputs/orderParameter/L_{L}/*.tsv')

        p_list = []
        xspan_list = []

        for file in files:
            p = float(file.split('_')[-1].rsplit('.tsv')[0])
            step, xspan, yspan = np.loadtxt(file, unpack=True, delimiter='\t', skiprows=1)
            p_list.append(p)
            xspan_list.append(np.mean(xspan))

        # sort p_list and the other lists accordingly
        p_list, xspan_list = zip(*sorted(zip(p_list, xspan_list)))
        xspan_vs_L_list.append(xspan_list)

    set_matplotlib_params()
    plt.figure(figsize=(8, 5))
    markers = ['^', 's', 'o']
    for L, xspan, marker in zip(L_list, xspan_vs_L_list, markers):
        plt.plot(p_list, xspan, marker=marker, linestyle='--', linewidth=1, label=f'L={L}')
    plt.xlabel('p')
    plt.ylabel('$R^{(h)}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'src/dp_paper/plots/orderParameter/system_size_comparison.png', dpi=300)
    plt.show()




if __name__ == "__main__":
    # main()
    systemSizeComparison()