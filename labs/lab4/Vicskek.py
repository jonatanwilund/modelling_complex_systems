import numpy as np
import matplotlib.pyplot as plt

N = 500
R = 0.1
v0 = 0.03
T = 1000
eta = 2 * np.pi

np.random.seed(1)


def initialize_particles(N):
    positions = np.random.rand(N, 2)
    theta = 2 * np.pi * np.random.rand(N)

    return positions, theta


def periodic_displacement(pos_i, positions):
    disp = positions - pos_i
    disp = disp - np.round(disp)

    return disp


def vicsek_step(positions, theta, R, eta, v0):
    N = len(theta)
    theta_new = np.zeros(N)

    for i in range(N):
        disp = periodic_displacement(positions[i], positions)

        distances = np.sqrt(np.sum(disp**2, axis=1))

        neighbors = distances < R

        mean_direction = np.sum(np.exp(1j * theta[neighbors]))

        theta_new[i] = np.angle(mean_direction) + eta * (np.random.rand() - 0.5)

        positions[:, 0] += v0 * np.cos(theta_new)
        positions[:, 1] += v0 * np.sin(theta_new)

        positions %= 1.0

    return positions, theta_new


def order_parameter(theta):
    mean_velocity = np.mean(np.exp(1j * theta))
    Phi = np.abs(mean_velocity)

    return Phi


def run_simulation(N, R, eta, v0, T):
    positions, theta = initialize_particles(N)
    position_history = np.zeros(shape=(T + 1, positions.shape[0], positions.shape[1]))
    position_history[0] = positions
    theta_history = np.zeros(shape=(T + 1, np.size(theta)))
    theta_history[0] = theta

    Phi_values = np.zeros(T + 1)
    Phi_values[0] = order_parameter(theta)

    for t in range(1, T + 1):
        positions, theta = vicsek_step(positions, theta, R, eta, v0)
        position_history[t] = positions
        theta_history[t] = theta

        Phi_values[t] = order_parameter(theta)

    return position_history, theta_history, Phi_values


def plot_snapshot(position_history, theta_history, eta, t):
    positions = position_history[t]
    theta = theta_history[t]

    plt.figure(figsize=(6, 6))

    plt.quiver(
        positions[:, 0],
        positions[:, 1],
        np.cos(theta),
        np.sin(theta),
        angles="xy",
        scale_units="xy",
        scale=25,
        width=0.003,
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        fr"Vicsek model snapshot for N = {N} particles at t = {t}, eta = 2$\pi$, \n R = {R}, $v_0$ = {v0}"
    )

    plt.savefig(f"snapshot_eta={eta}_t={t}.png")

    plt.show()


def plot_phi(Phi_values, eta):

    time = np.arange(len(Phi_values))

    plt.figure(figsize=(7, 5))

    plt.plot(time, Phi_values)

    plt.xlabel("Time step t")
    plt.ylabel(r"$\Phi(t)$")

    plt.title(fr"Order parameter vs time, eta = 2$\pi$")

    plt.grid(True)

    plt.savefig(f"Phi_vs_time_eta=2pi.png")

    plt.show()


def time_average_phi(Phi_values, transient_fraction=0.5):

    start_index = int(transient_fraction * len(Phi_values))

    return np.mean(Phi_values[start_index:])


def plot_average_phi_vs_eta(N, R, v0, T):

    eta_values = np.linspace(0, 2 * np.pi, 20)

    average_phi_values = np.zeros(len(eta_values))

    for k, eta in enumerate(eta_values):
        print(f"Running eta = {eta:.3f}")

        _, _, Phi_values = run_simulation(N, R, eta, v0, T)

        average_phi_values[k] = time_average_phi(Phi_values)

    plt.figure(figsize=(7, 5))

    plt.plot(eta_values, average_phi_values, marker="o")

    plt.xlabel(r"Noise strength $\eta$")
    plt.ylabel(r"Time-averaged $\Phi$")

    plt.title("Vicsek model: average alignment vs noise")

    plt.grid(True)
    plt.savefig("time_avg_Phi_vs_eta.png")

    plt.show()


plot_average_phi_vs_eta(N, R, v0, T)
