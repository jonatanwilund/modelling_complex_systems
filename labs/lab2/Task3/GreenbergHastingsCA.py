import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import initconfigs

def save_grid(grid, filename):
    np.savetxt(filename, grid, fmt="%d")

def load_grid(filename):
    return np.loadtxt(filename, dtype=np.uint8)

def find_period(grid, N, e, max_steps=10000): 
    """
    This function finds the transient time and eventual period of a GHCA orbit. 

    -----
    Args: 
    -----

    grid: np.ndarray. 
    Initial configuration grid. 

    e: int. 
    Excitation parameter.
    
    N: int
    number of states

    max_steps: int. 
    Max number of evolutions to detect period in. 

    --------
    Returns: 
    --------

    transient: int. 
    Number of steps before the repeating cycle starts. 

    period: int. 
    Length of the repeating cycle. 
    """

    seen = {}

    current = grid.astype(np.uint8).copy()

    for t in range(max_steps + 1): 
        key = current.tobytes()

        if key in seen: # As soon as we detect Xt = Xs
            transient = seen[key] # The transient is s
            period = t - seen[key] # And the period would be t-s
            return transient, period
    
        seen[key] = t
        current = GHCA_step(current, N, e)

    # Using runtimeerror for smoother experimenting later on
    raise RuntimeError("No period found within max_steps.") 


def GHCA_step(grid, N, e): 
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
    excited_mask = (grid >= 1) & (grid <= e)

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
    next_grid[(grid >= 1) & (grid < N - 1)] = grid[(grid >= 1) & (grid < N - 1)] + 1

    # Rule 3: Cells in state N-1 return to resting 0.
    # This is handled since the new grid was initialized to 0 everywhere! 

    return next_grid

def random_grid(n, N, e, seed=42): 
    """
    This function generates a random grid of size n x n to be used as an initial configuration for the GHCA.

    -----
    Args: 
    -----

    n: int. 
    Grid size is n x n.

    e: int. 
    Excitation parameter. 
    
    N: int
    Number of states
    
    seed: int or None. 
    For reproducibility. Defaults to 42, of course.

    --------
    Returns:
    --------

    rand_grid: np.ndarray. 
    An array of size n x n with random integer entries between 0 and N-1

    """

    RNG = np.random.default_rng(seed)
    rand_grid = RNG.integers(low=0, high=N, size=(n,n), dtype=np.uint8)

    return rand_grid

def run_GHCA(grid, N, e, k): 
    """
    This function runs the actual evolution of the GHCA for k time steps or "generations".

    -----
    Args: 
    -----

    grid: np.ndarray. 
    The initial n x n grid with random configuration

    e: int. 
    Excitation parameter
    
    N: int
    Number of states
    
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
        current = GHCA_step(current, N, e)
        history[i+1] = current

    return history

def static_plot_grid(grid, N, e, savefigformat=''): 
    """
    Plots one single GHCA configuration. 

    -----
    Args: 
    -----

    grid: np.ndarray.
    Grid to be plotted.

    e: int
    Excitation parameter.
    
    N: int
    Number of states

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
        "red",      
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

    cmap = ListedColormap(colors[:N])

    fig, ax = plt.subplots()

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=N-1)
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

def animate_evolution(history, N, e, interval=200): 
    """
    This function animates the GHCA evolution. 

    -----
    Args: 
    -----

    history: np.ndarray.
    An array containing all states of the GHCA. This is an array of grids. 

    e: int. 
    Excitation parameter.
    
    N: int
    Number of states.

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

    cmap = ListedColormap(colors[:N])

    fig, ax = plt.subplots()

    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=N-1)

    n = history.shape[1]

    if n <= 200: 
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.1)

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
        blit=False
    )

    return animatn


if __name__ == "__main__": 
    n = 100
    N = 17
    e = 10
    k = 200
    max_steps = 10000
    seed = 42


    random_initial = random_grid(n, N, e, seed=seed)
    broken_circle = initconfigs.broken_circle_grid(n, N)
    single_seed_grid = initconfigs.single_seed_grid(n, N)
    planar_wave = initconfigs.planar_wave_grid(n,N)
    diagwave_initial = initconfigs.diagonal_wave_grid(n, N)
    multseed_initial = initconfigs.multiple_seed_grid(n, N)
    ringtail_initial = initconfigs.ring_with_tail_grid(n, N)
    collisionwaves_initial = initconfigs.collision_waves_grid(n, N)
    concrings_initial = initconfigs.concentric_rings_grid(n, N)
    funlarge_initial = initconfigs.fun_large_grid(n, N)

    history = run_GHCA(funlarge_initial, N, e, k)

    animatn = animate_evolution(history, N, e)
    
    plt.show()
    #save_grid(history[-1], f'{k}th_config_{n}_{N}_{e}')
    #animatn.save("ghca.gif", writer="pillow", fps=10)

    #static_plot_grid(history[-1], e, savefigformat='png')


    
    #-----------------PERIOD-FINDING--------------: 
    transient, period = find_period(funlarge_initial, N, e, max_steps=max_steps)
    print(f"n = {n}")
    print(f"N = {N}")
    print(f"e = {e}")
    print(f"seed = {seed}")
    print(f"transient time = {transient}")
    print(f"period = {period}")
    



   
