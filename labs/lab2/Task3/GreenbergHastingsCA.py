import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import initconfigs

def save_grid(grid, filename):
    np.savetxt(filename, grid, fmt="%d")

def load_grid(filename):
    return np.loadtxt(filename, dtype=np.uint8)


def GHCA_step(grid, e): 
    """
    This function takes a grid and evolves it one generation. 

    !!!It is in this function all the GHCA rules are applied!!!

    Note: We use four nearest neighbors, i.e Von Neumann neighborhood. 
    up down left right. 

    -----
    Args:
    -----

    grid: np.ndarray. 
    An nxn array with integer states 0 to e.

    e: int. 
    Excitation parameter.

    --------
    Returns:
    --------

    next_grid: np.ndarray. 
    The updated grid
    """

    # Boolean mask: True where excited
    excited_mask = (grid == 1)

    # Check whether each cell has an excited neighbor: 
    excited_neighbor = (
        np.roll(excited_mask, shift=1, axis=0)  |   # top neighbor
        np.roll(excited_mask, shift=-1, axis=0) |   # bottom neighbor
        np.roll(excited_mask, shift=1, axis=1)  |   # left neighbor
        np.roll(excited_mask, shift=-1, axis=1)     # right neighbor
    )

    next_grid = np.zeros_like(grid, dtype=np.uint8)

    # Rule 1: Resting cells become excited if at least one neighbor is excited. 
    # Puts on a mask that picks all cells that are resting AND that have excited neighbors:
    next_grid[(grid == 0) & excited_neighbor] = 1

    # Rule 2: Excited/recovering cells advance by one state (+1).
    # Puts on a mask that picks out all cells that are between 1 and e and and increments by one:
    next_grid[(grid >= 1) & (grid < e)] = grid[(grid >= 1) & (grid < e)] + 1

    # Rule 3: Cells in state e return to resting 0.
    # This is handled since the new grid was initialized to 0 everywhere! 

    return next_grid

def random_grid(n, e, seed=42): 
    """
    This function generates a random grid of size n x n to be used as an initial configuration for the GHCA.

    -----
    Args: 
    -----

    n: int. 
    Grid size is n x n.

    e: int. 
    Excitation parameter. 

    seed: int or None. 
    For reproducibility. Defaults to 42, of course.

    --------
    Returns:
    --------

    rand_grid: np.ndarray. 
    An array of size n x n with random integer entries between 0 and e

    """

    RNG = np.random.default_rng(seed)
    rand_grid = RNG.integers(low=0, high=e+1, size=(n,n), dtype=np.uint8)

    return rand_grid

def run_GHCA(grid, e, k): 
    """
    This function runs the actual evolution of the GHCA for k time steps or "generations".

    -----
    Args: 
    -----

    grid: np.ndarray. 
    The initial n x n grid with random configuration

    e: int. 
    Excitation parameter

    k: int. 
    Number of generations
    
    --------
    Returns:
    --------

    history: np.ndarray. 
    All k+1 grids, or generations. 

    """

    current = grid.astype(np.uint8).copy()
    n = grid.shape[0]
    history = np.zeros((k+1, n, n), dtype=np.uint8) 
    history[0] = current

    for i in range(k): 
        current = GHCA_step(current, e)
        history[i+1] = current

    return history

def static_plot_grid(grid, e, savefigformat=''): 
    """
    Plots one single GHCA configuration. 

    -----
    Args: 
    -----

    grid: np.ndarray.
    Grid to be plotted.

    e: int
    Excitation parameter.

    savefigformat: String. 
    Optional argument, if added the function saves the plot in the specified file extension format. 
    Defaults to empty string and does thereby not save the plot. 


    --------
    Returns:
    --------

    None 

    """

    colors = [
        "white",    # 0 = resting
        "red",      # 1 = excited
        "orangered",
        "darkorange", 
        "orange", 
        "gold",
        "yellow",
        "greenyellow",
        "chartreuse", 
        "lawngreen",
        "forestgreen",
        "green",  
        "darkcyan",
        "blue",   
        "slateblue", 
        "blueviolet",
        "darkviolet",
        "purple", 
        "mediumvioletred", 
        "magenta", 
        "deeppink", # 21
    ]

    cmap = ListedColormap(colors[:e+1])

    fig, ax = plt.subplots()

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=e)
    n = grid.shape[0]

    if n <= 200: 
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.1)

    ax.set_xticks([])
    ax.set_yticks([])

    if savefigformat: 
        plt.savefig(f'plot_{k}th_config_{n}_{e}.{savefigformat}')
    plt.show()

def animate_evolution(history, e, interval=200): 
    """
    This function animates the GHCA evolution. 

    -----
    Args: 
    -----

    history: np.ndarray.
    An array containing all states of the GHCA. This is an array of grids. 

    e: int. 
    Excitation parameter. 

    interval: int. 
    Time between frames in milliseconds.

    --------
    Returns:
    --------

    None

    """

    colors = [
        "white",    # 0 = resting
        "red",      # 1 = excited
        "orangered",
        "darkorange", 
        "orange", 
        "gold",
        "yellow",
        "greenyellow",
        "chartreuse", 
        "lawngreen",
        "forestgreen",
        "green",  
        "darkcyan",
        "blue",   
        "slateblue", 
        "blueviolet",
        "darkviolet",
        "purple", 
        "mediumvioletred", 
        "magenta", 
        "deeppink", # 21
    ]

    cmap = ListedColormap(colors[:e+1])

    fig, ax = plt.subplots()

    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=e)

    n = history.shape[1]

    if n <= 200: 
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame): 
        im.set_array(history[frame])
        return [im]
    
    animatn = animation.FuncAnimation(
        fig, 
        update, 
        frames=history.shape[0],
        interval=interval,
        blit=True
    )

    plt.show()


if __name__ == "__main__": 
    n = 201
    e = 7
    k = 200

    random_initial = random_grid(n, e)
    #broken_circle = initconfigs.broken_circle_grid(n, e)
    history = run_GHCA(random_initial, e, k)
    #animate_evolution(history, e)

    save_grid(history[-1], f'{k}th_config_{n}_{e}')

    static_plot_grid(history[-1], e, savefigformat='png',)
   

    """
    We have to: S
    save last grid to a file ***DONE
    plot last grid: 

    b) write code that detects periodicity
    c) find initial condition that yields period m >= 2
    """
