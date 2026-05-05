import numpy as np


def broken_circle_grid(n, e, R=None, thickness=1.5, gap_angle=np.pi/4):
    """
    Create an initial condition with a broken circular wavefront.

    Parameters
    ----------
    n : int
        Grid size (n x n)

    e : int
        Excitation parameter

    R : float or None
        Radius of the circle. If None, set automatically.

    thickness : float
        Thickness of the ring

    gap_angle : float
        Angular size of the gap (in radians)

    Returns
    -------
    grid : np.ndarray
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    cx, cy = n // 2, n // 2

    if R is None:
        R = n // 4

    # Create coordinate arrays
    y, x = np.indices((n, n))

    dx = x - cx
    dy = y - cy

    r = np.sqrt(dx**2 + dy**2)

    # Ring mask
    ring = (R - thickness <= r) & (r <= R + thickness)

    # Angle (range: -pi to pi)
    theta = np.arctan2(dy, dx)

    # Remove a sector (gap)
    gap = np.abs(theta) < gap_angle

    broken_ring = ring & (~gap)

    # Set excited cells
    grid[broken_ring] = 1

    return grid