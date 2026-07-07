import numpy as np
import matplotlib.pyplot as plt
import csv


def is_column_stochastic(P, tol=1e-12):
    nonnegative = np.all(P >= -tol)
    col_sums = np.sum(P, axis=0)
    columns_sum_to_one = np.allclose(col_sums, 1.0, atol=tol)

    return nonnegative and columns_sum_to_one


def stationary_distribution(P):
    eigvals, eigvecs = np.linalg.eig(P)

    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(
        eigvecs[:, idx]
    )  # Using np.real() since there can be small unwanted imaginary parts

    # linalg.eig can arbitrarily return v or -v. We want the one with positive elements, so:
    if np.sum(pi) < 0:
        pi = -pi

    pi = (
        pi / sum(pi)
    )  # Normalize with L1 norm (here it actually is L1 bc of PF-theorem) to ensure elements sum to 1

    return pi


# !!! Might be problematic to just blindly pick the second-largest eigval bc of numerical flp stuff
def second_largest_eigenvalue_modulus(P):
    eigvals = np.linalg.eigvals(P)

    moduli = np.sort(np.abs(eigvals))
    mu = moduli[-2]
    gap = 1 - mu

    return mu, gap


def total_variation_distance(x, pi):
    return 0.5 * np.sum(np.abs(x - pi))


def simulate_convergence(P, pi, x0, T):
    distances = np.zeros(T + 1)
    x = x0.copy()

    for t in range(T + 1):
        distances[t] = total_variation_distance(x, pi)
        x = P @ x

    return distances


def main():
    P0 = np.array(
        [
            [0.60, 0.20, 0.15, 0.10],
            [0.20, 0.50, 0.20, 0.20],
            [0.10, 0.20, 0.45, 0.20],
            [0.10, 0.10, 0.20, 0.50],
        ]
    )

    matrices = {
        "P_0": P0,
        "P_0.5": 0.5 * np.eye(4) + 0.5 * P0,
        "P_0.8": 0.8 * np.eye(4) + 0.2 * P0,
    }

    T = 100
    convergence_tol = 1e-3
    x0 = np.array([1.0, 0.0, 0.0, 0.0])

    plt.figure(figsize=(8, 5))

    table_rows = []

    for label, P in matrices.items():
        print("-" * len(label))
        print(f"{label}")
        print("-" * len(label))

        print(f"Column-stochastic: {is_column_stochastic(P)}")

        pi = stationary_distribution(P)
        print(f"Stationary distribution: pi = {pi}")
        print(f" 0 if pi is actually stationary dist: {np.linalg.norm(P @ pi - pi)}")

        mu, gap = second_largest_eigenvalue_modulus(P)
        print(f"Eigenvalues = {np.linalg.eigvals(P)}")
        print(f"mu = {mu}")
        print(f"Spectral gap = {gap}")

        distances = simulate_convergence(P, pi, x0, T)
        below = np.where(distances < convergence_tol)[0]
        if len(below) > 0:
            observed_time = int(below[0])
        else:
            observed_time = None

        table_rows.append((label, mu, gap, observed_time))

        t = np.arange(T + 1)
        plt.semilogy(
            t,
            distances,
            label=f"{label}: Tot. var. distance",
        )

        comparison = distances[0] * (mu**t)
        plt.semilogy(
            t, comparison, "--", label=f"{label}: " + r"proportional to $\mu^t$"
        )

    plt.xlabel("t")
    plt.ylabel(r"Total variation distance $\delta(t)$")
    plt.title("Convergence to stationary distribution")
    plt.grid(True)
    plt.legend()
    plt.savefig("markov_convergence.png")
    plt.show()

    with open("markov_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Matrix", "Mu", "Spectral gap", "Steps to convergence"])

        for label, mu, gap, observed_time in table_rows:
            writer.writerow([label, mu, gap, observed_time])


if __name__ == "__main__":
    main()


"""
AI/LLM STATEMENT: 

This code was developed with the continuous consultation of ChatGPT 5.5. ChatGPT has 
mostly helped with questions regarding the usage of the various numpy-functions in this 
code, as well as of the csv module. This was done for speed, since it is faster than reading 
through the official documentations. Any and all content and functionality of this code
is claimed as defensible by its developer. 
"""
