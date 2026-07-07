import numpy as np


def _clip_state(value, N):
    """Return an integer state clipped to the allowed range 0, ..., N-1."""
    return np.uint8(np.clip(value, 0, N - 1))


def _coordinate_grid(n):
    """Return y, x coordinate arrays for an n x n grid."""
    return np.indices((n, n))


def broken_circle_grid(n, N, R=None, thickness=1.5, gap_angle=np.pi / 4):
    """
    Create an initial condition with a broken circular wavefront.

    Parameters
    ----------
    n : int
        Grid size (n x n).

    N : int
        Size of state space

    R : float or None
        Radius of the circle. If None, set automatically.

    thickness : float
        Thickness of the ring.

    gap_angle : float
        Angular size of the gap in radians.

    Returns
    -------
    grid : np.ndarray
        Initial n x n configuration.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    cx, cy = n // 2, n // 2

    if R is None:
        R = n // 4

    y, x = _coordinate_grid(n)

    dx = x - cx
    dy = y - cy

    r = np.sqrt(dx**2 + dy**2)
    ring = (R - thickness <= r) & (r <= R + thickness)

    theta = np.arctan2(dy, dx)
    gap = np.abs(theta) < gap_angle

    broken_ring = ring & (~gap)
    grid[broken_ring] = 1

    return grid


def single_seed_grid(n, N, center=None):
    """
    One excited state 1 cell in an otherwise resting grid.
    Useful for observing radial wave propagation.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    if center is None:
        center = (n // 2, n // 2)

    cy, cx = center
    grid[cy % n, cx % n] = 1

    return grid


def multiple_seed_grid(n, N, centers=None, radius=2):
    """
    Several small excited state 1 disks in an otherwise resting grid.
    Useful for observing several wavefronts and collisions.
    """

    grid = np.zeros((n, n), dtype=np.uint8)
    y, x = _coordinate_grid(n)

    if centers is None:
        centers = [
            (n // 4, n // 4),
            (n // 4, 3 * n // 4),
            (3 * n // 4, n // 4),
            (3 * n // 4, 3 * n // 4),
            (n // 2, n // 2),
        ]

    for cy, cx in centers:
        disk = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        grid[disk] = 1

    return grid


def planar_wave_grid(n, N, width=3, direction="vertical", position=None):
    """
    A straight excited state 1 line.
    direction='vertical' gives a vertical line; direction='horizontal' gives a horizontal line.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    if position is None:
        position = n // 3

    half_width = width // 2

    if direction == "vertical":
        cols = np.arange(position - half_width, position + half_width + 1) % n
        grid[:, cols] = 1
    elif direction == "horizontal":
        rows = np.arange(position - half_width, position + half_width + 1) % n
        grid[rows, :] = 1
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'.")

    return grid


def diagonal_wave_grid(n, N, width=3, offset=0):
    """
    A diagonal excited wavefront of approximate form y - x = offset.
    """

    grid = np.zeros((n, n), dtype=np.uint8)
    y, x = _coordinate_grid(n)

    diagonal = np.abs(y - x - offset) <= width
    grid[diagonal] = 1

    return grid


def diagonal_wave_with_tail_grid(n, N, width=3, offset=0, tail_spacing=8):
    """
    A diagonal wavefront with refractory bands behind it.
    This often gives a more visually interpretable traveling front than a single line.
    """

    grid = np.zeros((n, n), dtype=np.uint8)
    y, x = _coordinate_grid(n)

    front = np.abs(y - x - offset) <= width
    grid[front] = 1

    if N >= 2:
        tail_states = [
            _clip_state(round(N / 3), N),
            _clip_state(round(2 * N / 3), N),
            _clip_state(N, N),
        ]

        for layer, state in enumerate(tail_states, start=1):
            band = np.abs(y - x - offset - layer * tail_spacing) <= width
            grid[band] = state

    return grid


def ring_with_tail_grid(n, N, center=None, R=None, thickness=2.0, tail_width=5.0):
    """
    A circular excited wavefront with refractory rings behind it.
    The front is state 1; inner rings are recovery states.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    if center is None:
        center = (n // 2, n // 2)

    if R is None:
        R = n // 4

    cy, cx = center
    y, x = _coordinate_grid(n)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    front = (R - thickness <= r) & (r <= R + thickness)
    grid[front] = 1

    if N >= 2:
        tail_states = [
            _clip_state(round(N / 3), N),
            _clip_state(round(2 * N / 3), N),
            _clip_state(N, N),
        ]

        for layer, state in enumerate(tail_states, start=1):
            inner_outer = R - thickness - (layer - 1) * tail_width
            inner_inner = R - thickness - layer * tail_width
            tail = (inner_inner <= r) & (r < inner_outer)
            grid[tail] = state

    return grid


def concentric_rings_grid(n, N, center=None, radii=None, thickness=2.0):
    """
    Several excited circular wavefronts.
    Useful for creating many wave collisions.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    if center is None:
        center = (n // 2, n // 2)

    if radii is None:
        radii = [n // 8, n // 5, n // 3]

    cy, cx = center
    y, x = _coordinate_grid(n)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    for R in radii:
        ring = (R - thickness <= r) & (r <= R + thickness)
        grid[ring] = 1

    return grid


def collision_waves_grid(n, N, width=3, margin=None):
    """
    Two opposing planar waves placed on the left and right sides of the domain.
    Useful for demonstrating wave collision and annihilation.
    """

    grid = np.zeros((n, n), dtype=np.uint8)

    if margin is None:
        margin = n // 4

    half_width = width // 2

    left_cols = np.arange(margin - half_width, margin + half_width + 1) % n
    right_cols = np.arange(n - margin - half_width, n - margin + half_width + 1) % n

    grid[:, left_cols] = 1
    grid[:, right_cols] = 1

    return grid


def sparse_random_excitation_grid(n, N, probability=0.01, seed=42):
    """
    Mostly resting grid with sparse excited cells.
    This gives cleaner waves than a fully random grid over all states.
    """

    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=np.uint8)

    excited = rng.random((n, n)) < probability
    grid[excited] = 1

    return grid


def random_patch_grid(n, N, patch_radius=None, probability=0.35, seed=42, center=None):
    """
    Random excited cells inside a circular patch, with the rest of the grid resting.
    Useful for generating asymmetric wave interactions near the center.
    """

    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=np.uint8)

    if center is None:
        center = (n // 2, n // 2)

    if patch_radius is None:
        patch_radius = n // 5

    cy, cx = center
    y, x = _coordinate_grid(n)

    patch = (x - cx) ** 2 + (y - cy) ** 2 <= patch_radius**2
    random_excited = rng.random((n, n)) < probability

    grid[patch & random_excited] = 1

    return grid


def fun_large_grid(n, N, seed=12):
    """
    A visually rich large-domain initial condition.
    Combines circular fronts, refractory tails, a diagonal front, and sparse noise.
    Suitable for larger n and larger N.
    """

    grid = np.zeros((n, n), dtype=np.uint8)
    y, x = _coordinate_grid(n)

    centers = [
        (n // 4, n // 4),
        (3 * n // 4, n // 4),
        (n // 4, 3 * n // 4),
        (3 * n // 4, 3 * n // 4),
        (n // 2, n // 2),
    ]

    ring_radius = max(8, n // 28)
    tail_width = max(4, n // 200)

    for cy, cx in centers:
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        grid[(r > ring_radius) & (r < ring_radius + tail_width)] = 1

        if N >= 2:
            grid[(r > ring_radius - tail_width) & (r <= ring_radius)] = _clip_state(
                N // 3, N
            )
            grid[
                (r > ring_radius - 2 * tail_width) & (r <= ring_radius - tail_width)
            ] = _clip_state(2 * N // 3, N)
            grid[
                (r > ring_radius - 3 * tail_width) & (r <= ring_radius - 2 * tail_width)
            ] = _clip_state(N, N)

    diagonal_width = max(2, n // 350)
    diagonal = np.abs(y - x) <= diagonal_width
    grid[diagonal] = 1

    if N >= 2:
        spacing = max(6, n // 125)
        grid[np.abs(y - x - spacing) <= diagonal_width] = _clip_state(N // 3, N)
        grid[np.abs(y - x - 2 * spacing) <= diagonal_width] = _clip_state(2 * N // 3, N)
        grid[np.abs(y - x - 3 * spacing) <= diagonal_width] = _clip_state(N, N)

    rng = np.random.default_rng(seed)
    noise = rng.random((n, n)) < 0.0008
    grid[noise] = 1

    return grid
