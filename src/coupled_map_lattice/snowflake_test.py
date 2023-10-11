import matplotlib.pyplot as plt
import numpy as np

def visualize_hex_lattice(hex_lattice):
    '''
    Visualizes a hexagonal lattice.
    
    Parameters
    ----------
    hex_lattice : np.ndarray
        The hexagonal lattice.
        
    Returns
    -------
        None
    '''
    x = hex_lattice.shape[0]
    y = hex_lattice.shape[1]
    xv, yv = np.meshgrid(np.linspace(0, x, x), np.arange(y), sparse=False, indexing='xy')
    xv[::2, :] += 0.5
    plt.scatter(xv, yv, c=hex_lattice)
    plt.show()


def visualize_neighbours(L):
    '''
    Toy function to visualize a lattice of size L, with a point and it's neighbours coloured in
    
    Parameters
    ----------
    L : int
        The side length of the lattice.
    
    Returns
    -------
        None
    '''
    lattice = np.zeros((L, L))

    lattice[5, 5] = 1
    lattice[5, 4] = 0.5
    lattice[5, 6] = 0.5
    lattice[4, 5] = 0.5
    lattice[6, 5] = 0.5
    lattice[4, 4] = 0.5
    lattice[6, 4] = 0.5

    visualize_hex_lattice(lattice)


def update(ice_lattice, temp_lattice, T_c):
    '''
    Updates the lattice according to the snowflake model.
    
    Parameters
    ----------
    ice_lattice : np.ndarray
        The ice lattice.
    temp_lattice : np.ndarray
        The temperature lattice.
    T_c : float
        The critical temperature.
        
    Returns
    -------
        None
    '''
    ice_lattice[temp_lattice < T_c] = 1
    



def main():
    L = 10
    
    ice_lattice = np.zeros((L, L))
    temp_lattice = np.ones((L, L))

    temp_lattice[L//2, L//2] = 0
    T_c = 0.9  # Critical temperature




    visualize_neighbours(L)

if __name__ == '__main__':
    main()
