import numpy as np
import matplotlib.pyplot as plt


def logistic_map(r, x):
    return r * x * (1 - x)


def step(r1, r2, eps, x, y):
    x_next = (1 - eps) * logistic_map(r1, x) + eps * logistic_map(r2, y)
    y_next = (1 - eps) * logistic_map(r2, y) + eps * logistic_map(r1, x)
    return x_next, y_next


def main():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    params = [
        (2.8, 2.9, 0.1, ax1),  # r1, r2, epsilon, subplot axis
        (3.1, 3.4, 0.3, ax2),
        (3.85, 3.95, 0.2, ax3),
    ]

    n_transient = 1024  # Number of transient iterations to discard
    n_record = 1024

    for r1, r2, eps, ax in params:
        x, y = 0.1, 0.6

        for _ in range(n_transient):
            x, y = step(r1, r2, eps, x, y)

        trajectory = []

        for _ in range(n_record):
            x, y = step(r1, r2, eps, x, y)
            trajectory.append((x, y))

        trajectory = np.array(trajectory)

        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            label=f"r1={r1}, r2={r2}, eps={eps}",
            color="tab:orange",
            marker="o",
            alpha=0.6,
            edgecolor="k",
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("x")
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend()

    fig.supylabel("y")
    fig.suptitle("Phase Portrait of Coupled Logistic Maps")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    main()
