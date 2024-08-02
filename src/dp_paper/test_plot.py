import numpy as np
import matplotlib.pyplot as plt

def set_matplotlib_params():

    colors = ['#901A1E', '#16BAC5', '#666666']

    plt.rcParams.update({'font.size': 16, 
                        #  'font.family': 'MS Reference Sans Serif', 
                         'figure.figsize': (14, 6), 
                         'axes.grid': True, 
                         'axes.prop_cycle': plt.cycler('color', colors),  # Set the color cycle
                         'grid.linestyle': '-',
                         'grid.linewidth': 0.5, 
                         'grid.alpha': 0.8, })

def main():
    set_matplotlib_params()
    x = np.geomspace(0.01, 100, 1000)
    y = np.sin(1/x)
    plt.plot(x, y, label='sin(x)')
    plt.plot(x, 1/x, label='1/x')
    plt.plot(x, 1/x**2, label='1/x^2')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()