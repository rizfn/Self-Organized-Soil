import numpy as np
import matplotlib.pyplot as plt
from numba import njit
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

@njit
def handle_overflow(density_lattice, worm_lattice, interaction_factor):
    '''
    In cases where the density lattice is > 1, this function will redistribute the excess to neighbouring cells.

    Not called directly, always called by `interact_worm_soil`.

    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    worm_lattice : np.ndarray
        The worm lattice.
    interaction_factor : float
        The speed at which the density lattice will approach the worm lattice. Max 1, where a cell gives everything to it's neighbours.

    Returns
    -------
        None
    '''
    L = density_lattice.shape[0]
    while np.max(density_lattice) > 1:
        overflowing_cells = np.argwhere(density_lattice > 1)
        for i in range(len(overflowing_cells)):
            c = overflowing_cells[i]
            overflow_amount = interaction_factor * density_lattice[c[0], c[1]] * worm_lattice[c[0], c[1]] / 4
            density_lattice[c[0], c[1]] -= 4 * overflow_amount
            density_lattice[(c[0]-1) % L, c[1]] += overflow_amount
            density_lattice[(c[0]+1) % L, c[1]] += overflow_amount
            density_lattice[c[0], (c[1]-1) % L] += overflow_amount
            density_lattice[c[0], (c[1]+1) % L] += overflow_amount

    ## todo: find way to vectorize, something like below.
    # while np.max(density_lattice) > 1:
    #     mask = density_lattice > 1
    #     density_lattice[mask] += interaction_factor * (np.roll(density_lattice, 1, axis=0)[mask] * np.roll(worm_lattice, 1, axis=0)[mask] + np.roll(density_lattice, -1, axis=0)[mask] * np.roll(worm_lattice, -1, axis=0)[mask] + np.roll(density_lattice, 1, axis=1)[mask] * np.roll(worm_lattice, 1, axis=1)[mask] + np.roll(density_lattice, -1, axis=1)[mask] * np.roll(worm_lattice, -1, axis=1)[mask] - 4 * density_lattice[mask] * worm_lattice[mask]) / 4


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


def update_lattices(density_lattice, worm_lattice):  # todo: remove safeguards if not needed
    '''
    Updates the density and worm lattices. Called once per timestep.

    Smoothen, reproduce, interact.

    Parameters
    ----------
    density_lattice : np.ndarray
        The density lattice.
    worm_lattice : np.ndarray
        The worm lattice.

    Returns
    -------
        None
    '''
    smooth_density_lattice(density_lattice, 0.1)
    # if np.max(density_lattice > 1):
    #     print("ALERT! Smoothening, density lattice > 1")
    reproduce_worms(density_lattice, worm_lattice)
    # if np.max(worm_lattice > 1):
    #     print("ALERT! Reproducing, worm lattice > 1")
    interact_worm_soil(density_lattice, worm_lattice)
    # if np.max(density_lattice > 1):
    #     print("ALERT! Interacting, density lattice > 1")


def main():
    L = 100  # side length of the lattice
    n_steps = 1000  # number of steps to run the simulation


    density_lattice = (0.8 / 0.5) * np.random.rand(L, L) 
    worm_lattice = np.random.rand(L, L)

    # # DEBUGGING
    # density_lattice = np.zeros((L, L))
    # density_lattice[L//2, L//2] = 0.5
    # worm_lattice = np.ones((L, L))


    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(density_lattice, vmin=0, vmax=1)
    ax[0].title.set_text(f"{np.mean(density_lattice):.2f}, ({np.min(density_lattice):.2f}, {np.max(density_lattice):.2f})")
    ax[1].imshow(worm_lattice, vmin=0, vmax=1)
    ax[1].title.set_text(f"{np.mean(worm_lattice):.2f}, ({np.min(worm_lattice):.2f}, {np.max(worm_lattice):.2f})")


    for i in tqdm(range(n_steps)):
        update_lattices(density_lattice, worm_lattice)
        # set overall title to iteration number
        fig.suptitle(f"Step {i}")
        ax[0].cla()
        ax[0].imshow(density_lattice, vmin=0, vmax=1)
        ax[0].title.set_text(f"{np.mean(density_lattice):.2f}, ({np.min(density_lattice):.2f}, {np.max(density_lattice):.2f})")
        ax[1].cla()
        ax[1].imshow(worm_lattice, vmin=0, vmax=1)
        ax[1].title.set_text(f"{np.mean(worm_lattice):.2f}, ({np.min(worm_lattice):.2f}, {np.max(worm_lattice):.2f})")
        plt.pause(0.01)
    


if __name__ == '__main__':
    main()
