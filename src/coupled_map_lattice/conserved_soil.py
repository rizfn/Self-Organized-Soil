import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def smooth_density_lattice(density_lattice, smoothing_factor=0.1):
    '''
    Smooths the density lattice.

    Applies a mean filter with periodic boundaries, with neighbours reduced by smoothing_factor
    
    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    L : int
        The side length of the lattice.
    smoothing_factor : float
        The speed at which the density lattice will approach the average of its neighbours. 1 for equal weights (mean filter)

    Returns
    -------
        None
    '''
    density_lattice += smoothing_factor * (np.roll(density_lattice, 1, axis=0) + np.roll(density_lattice, -1, axis=0) + np.roll(density_lattice, 1, axis=1) + np.roll(density_lattice, -1, axis=1))
    density_lattice /= 1 + 4 * smoothing_factor

def target_worm_count(density_lattice):
    '''
    Returns the target worm count for a given density.
    
    If density lattice is 0.5, worms go to 1. if it's 0 or 1, worms go to 0. Assumed linear and symmetric on both sides of 0.5.
        
    Parameters
    ----------
    density : float
        The density of the lattice site.
        
    Returns
    -------
    worm_count : float
        The target worm count for the lattice site.
    '''
    target_lattice = np.zeros_like(density_lattice)
    target_lattice[density_lattice <= 0.5] = 2 * density_lattice[density_lattice <= 0.5]
    target_lattice[density_lattice > 0.5] = 2 - 2 * density_lattice[density_lattice > 0.5]
    target_lattice[density_lattice > 1] = 0
    return target_lattice

# def target_worm_count(density_lattice):
#     '''
#     Returns the target worm count for a given density.
    
#     Depends on the gradient of the density lattice. High |gradient|, worms go to 1, low |gradient|, worms go to 0.
        
#     Parameters
#     ----------
#     density : float
#         The density of the lattice site.
        
#     Returns
#     -------
#     worm_count : float
#         The target worm count for the lattice site.
#     '''
#     x_derivative = np.roll(density_lattice, 1, axis=0) - np.roll(density_lattice, -1, axis=0)
#     y_derivative = np.roll(density_lattice, 1, axis=1) - np.roll(density_lattice, -1, axis=1)
#     gradient = np.sqrt(x_derivative**2 + y_derivative**2)
#     gradient[gradient > 1] = 1
#     return gradient


def reproduce_worms(density_lattice, worm_lattice, birth_factor=0.1):
    '''
    Reproduces worms according to the target worm count.
    
    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    worm_lattice : np.ndarray
        The worm lattice.
    birth_factor : float
        The speed at which the worm lattice will approach the density lattice. Max 1.
        
    Returns
    -------
        None
    '''
    worm_lattice += birth_factor * (target_worm_count(density_lattice) - worm_lattice)

def interact_worm_soil(density_lattice, worm_lattice, interaction_factor=0.1):
    '''
    Interacts worms and soil according to the worm lattice.

    Each cell in the density lattice will donate it's values to a neighbouring cell, depending on the original cell's worm lattice value. A higher worm lattice value means a higher donation.
    
    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    worm_lattice : np.ndarray
        The worm lattice.
    L : int
        The side length of the lattice.
    interaction_factor : float
        The speed at which the density lattice will approach the worm lattice. Max 1, where a cell gives everything to it's neighbours.
        
    Returns
    -------
        None
    '''
    density_lattice += interaction_factor * (np.roll(density_lattice, 1, axis=0) * np.roll(worm_lattice, 1, axis=0) + np.roll(density_lattice, -1, axis=0) * np.roll(worm_lattice, -1, axis=0) + np.roll(density_lattice, 1, axis=1) * np.roll(worm_lattice, 1, axis=1) + np.roll(density_lattice, -1, axis=1) * np.roll(worm_lattice, -1, axis=1) - 4 * density_lattice * worm_lattice) / 4
    # interact again for any cells with density > 1
    # handle_overflow(density_lattice, worm_lattice, interaction_factor)


def update_lattices(density_lattice, worm_lattice, s, b, i):
    '''
    Updates the density and worm lattices. Called once per timestep.

    Smoothen, reproduce, interact.

    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    worm_lattice : np.ndarray
        The worm lattice.
    s : float
        The smoothening factor
    b : float
        The birth factor
    i : float
        The interaction factor

    Returns
    -------
        None
    '''
    smooth_density_lattice(density_lattice, s)
    reproduce_worms(density_lattice, worm_lattice, b)
    interact_worm_soil(density_lattice, worm_lattice, i)


def calculate_cluster_sizes(density_lattice):
    '''
    Calculates the cluster sizes for each timestep.
    
    Parameters
    ----------
    density_lattice : ndarray
        Array of density_lattice data for each timestep.
    
    Returns
    -------
    cluster_sizes : ndarray
        Array of cluster sizes for each timestep.
    '''
    from scipy import ndimage
    cluster_sizes = []
    m = np.round(density_lattice) > 0
    lw, num = ndimage.label(m)
    cluster_sizes = ndimage.sum(m, lw, index=np.arange(num + 1))
    return cluster_sizes


def main():
    L = 10000  # side length of the lattice
    n_steps = 1000  # number of steps to run the simulation
    initial_density = 0.8
    s_f = 0.6
    b_f = 0.1
    i_f = 1

    density_lattice = (initial_density / 0.5) * np.random.rand(L, L) 
    worm_lattice = np.random.rand(L, L)

    # fig, ax = plt.subplots(1, 2)

    # ax[0].imshow(density_lattice, vmin=0, vmax=1)
    # ax[0].title.set_text(f"{np.mean(density_lattice):.2f}, ({np.min(density_lattice):.2f}, {np.max(density_lattice):.2f})")
    # ax[1].imshow(worm_lattice, vmin=0, vmax=1)
    # ax[1].title.set_text(f"{np.mean(worm_lattice):.2f}, ({np.min(worm_lattice):.2f}, {np.max(worm_lattice):.2f})")


    for i in tqdm(range(n_steps)):
        update_lattices(density_lattice, worm_lattice, s_f, b_f, i_f)
        # # set overall title to iteration number
        # fig.suptitle(f"Step {i}")
        # ax[0].cla()
        # ax[0].imshow(density_lattice, vmin=0, vmax=1)
        # ax[0].title.set_text(f"{np.mean(density_lattice):.2f}, ({np.min(density_lattice):.2f}, {np.max(density_lattice):.2f})")
        # ax[1].cla()
        # ax[1].imshow(worm_lattice, vmin=0, vmax=1)
        # ax[1].title.set_text(f"{np.mean(worm_lattice):.2f}, ({np.min(worm_lattice):.2f}, {np.max(worm_lattice):.2f})")
        # plt.pause(0.01)


    # CLUSTER SIZES
    cluster_sizes = calculate_cluster_sizes(density_lattice)

    hist, edges = np.histogram(cluster_sizes, bins=100, density=True)
    plt.title(f'{L=}, {s_f=}, {b_f=}, {i_f=}')
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.plot(edges[:-1], hist, 'x', label='data')
    x = np.array(edges[:-1])
    plt.plot(x, 2*x**-2, label=r'$\tau=2$ power law')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"src/coupled_map_lattice/cluster_sizes_{L}_{s_f}_{b_f}_{i_f}.png", dpi=300)
    plt.show()
    


if __name__ == '__main__':
    main()
