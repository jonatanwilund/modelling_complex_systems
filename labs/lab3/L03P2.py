import numpy as np
import numpy.linalg as lg

P0 = np.array(
    [
        [0.6, 0.2, 0.15, 0.10],
        [0.2, 0.5, 0.2, 0.2],
        [0.10, 0.2, 0.45, 0.2],
        [0.1, 0.1, 0.2, 0.5],
    ]
)

col_sums = P0.sum(axis=0)

print(col_sums)

if np.allclose(col_sums, 1):
    print("columns sum to 1")
else:
    print("columns don't sum to 1")

eigenval, eigenvec = lg.eig(P0)

print(eigenval)


idx = np.argmin(np.abs(eigenval - 1))
pi = np.real(eigenvec[:, idx])
pi = pi / pi.sum()

print(pi)
print(P0 @ pi - pi)
print(pi.sum())
