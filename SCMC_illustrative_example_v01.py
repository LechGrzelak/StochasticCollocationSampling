import numpy as np
from scipy.stats import norm, gamma
from scipy.special import factorial2
import matplotlib.pyplot as plt

def build_moment_matrix(moment_fn, N):
    """
    Build (N+1) x (N+1) symmetric Hankel moment matrix M where
    M[i,j] = m_{i+j}, with 0-based indices mapping to moments 0..2N.

    Parameters
    ----------
    moment_fn : callable
        Function m(k) returning the k-th raw moment E[X^k].
    N : int
        Degree of the quadrature/interpolation minus 1 (number of nodes - 1).

    Returns
    -------
    M : ndarray, shape (N+1, N+1)
        Moment matrix.
    """
    M = np.empty((N + 1, N + 1), dtype=float)
    for i in range(N + 1):
        for j in range(N + 1):
            k = i + j
            # Special case to ensure exact parity with your MATLAB branch (i=j=0 => 1)
            # though for well-defined moment_fn, moment_fn(0) should already be 1
            M[i, j] = 1.0 if (i == 0 and j == 0) else moment_fn(k)
    return M

def gauss_quadrature_from_moment_matrix(M):
    """
    Given a (N+1)x(N+1) moment matrix M (assumed positive-definite),
    compute:
      - Recurrence coefficients (alpha, beta)
      - Collocation nodes x_i (eigenvalues of the Jacobi matrix)
      - Weights f_i = (first component of normalized eigenvectors)^2

    This mirrors the MATLAB pathway via Cholesky of M and the
    three-term recurrence for orthonormal polynomials.

    Returns
    -------
    x_i : ndarray, shape (N,)
        Collocation nodes (Gauss nodes).
    f_i : ndarray, shape (N,)
        Quadrature weights associated to x_i.
    alpha : ndarray, shape (N,)
        Recurrence alpha coefficients.
    beta : ndarray, shape (N-1,)
        Recurrence beta coefficients (positive).
    """
    # Cholesky: numpy returns lower-triangular L with L @ L.T = M
    # MATLAB chol returns upper-triangular R with R'.R = M.
    # Use R as upper triangle to keep the exact indexing used in your code.
    L = np.linalg.cholesky(M)
    R = L.T

    Np1 = M.shape[0]  # N+1
    N = Np1 - 1  # number of quadrature points

    alpha = np.zeros(N, dtype=float)
    beta = np.zeros(N - 1, dtype=float) if N > 1 else np.array([])

    # Python is 0-based:
    if N >= 1:
        alpha[0] = R[0, 1] / R[0, 0] * R[0, 0] / R[0, 0]  # same as R[0,1]; explicit for clarity
        if N > 1:
            beta[0] = (R[1, 1] / R[0, 0]) ** 2

    for i in range(1, N - 1):
        alpha[i] = R[i, i + 1] / R[i, i] - R[i - 1, i] / R[i - 1, i - 1]
        beta[i] = (R[i + 1, i + 1] / R[i, i]) ** 2

    if N >= 2:
        alpha[N - 1] = R[N - 1, N] / R[N - 1, N - 1] - R[N - 2, N - 1] / R[N - 2, N - 2]

    # Jacobi matrix (symmetric tridiagonal)
    J = np.diag(alpha)
    if N > 1:
        off = np.sqrt(beta)
        J += np.diag(off, k=1) + np.diag(off, k=-1)

    # Eigen-decomposition
    evals, evecs = np.linalg.eigh(J)
    x_i = evals
    f_i = (evecs[0, :]) ** 2  # first-row squares
    return x_i, f_i, alpha, beta

def lagrange_polynomial_values(y_i, x_i, X):
    """
    Evaluate Lagrange interpolant P(X) passing through (x_i, y_i).
    """
    y_i = np.asarray(y_i, dtype=float)
    x_i = np.asarray(x_i, dtype=float)
    X = np.asarray(X, dtype=float)

    N = len(y_i)
    L = np.ones((N, X.size), dtype=float)
    for i in range(N):
        for j in range(N):
            if j != i:
                L[i] *= (X - x_i[j]) / (x_i[i] - x_i[j])
    return y_i @ L

def ecdf(data):
    x = np.sort(np.asarray(data))
    n = x.size
    y = np.arange(1, n + 1) / n
    return y, x

def standard_normal_moment(k):
    """
    Raw moment of standard normal N(0,1):
      E[X^k] = 0 if k odd
      E[X^k] = (k-1)!! if k even
    """
    return float(factorial2(k - 1)) if (k % 2 == 0) else 0.0

def main_demo():
    np.random.seed(42)

    # Parameters
    NoOfPaths = 100000
    k_shape = 5
    theta = 2.0
    N = 4  # number of collocation points

    # 1) Build moment matrix from provided moment function
    M = build_moment_matrix(standard_normal_moment, N)

    # 2) Compute collocation nodes and weights from M
    x_i, f_i, alpha, beta = gauss_quadrature_from_moment_matrix(M)

    # Map nodes via exact CDF-to-quantile transform
    F_X = norm.cdf(x_i)
    y_i = gamma.ppf(F_X, a=k_shape, scale=theta)

    # Cheap normal samples & polynomial map g_N
    X = np.random.normal(0.0, 1.0, size=NoOfPaths)
    gN = lagrange_polynomial_values(y_i, x_i, X)

    # Exact gamma samples
    Y_exact = np.random.gamma(shape=k_shape, scale=theta, size=NoOfPaths)

    # --- Plot ECDFs ---
    y1, x1 = ecdf(Y_exact)
    y2, x2 = ecdf(gN)

    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, label='F_Y', linewidth=1.5)
    plt.plot(x2, y2, '--r', label='F_{g(X)}', linewidth=1.5)
    plt.legend()
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.xlabel('y')
    plt.ylabel('CDF(y)')
    plt.title('ECDF: Exact Gamma vs Polynomial Map g(X)')

    # Collocation lines (projected in y-CDF space)
    for yi, Fi in zip(y_i, F_X):
        plt.plot([yi, yi], [0, Fi], 'k:')
        plt.plot([0, yi], [Fi, Fi], 'k:')
    plt.plot(y_i, F_X, 'ok', markersize=6, markerfacecolor='black')
    plt.tight_layout()

    # --- Parametric plot ---
    plt.figure(figsize=(8, 5))
    U = np.linspace(1e-3, 1 - 1e-4, 200)
    X_grid = norm.ppf(U)
    Y_grid = gamma.ppf(U, a=k_shape, scale=theta)
    gN_grid = lagrange_polynomial_values(y_i, x_i, X_grid)

    plt.plot(X_grid, Y_grid, label='(X, Y)', linewidth=1.5)
    plt.plot(X_grid, gN_grid, '--r', label='(X, g(X))', linewidth=1.5)
    plt.plot(x_i, y_i, 'ok', markersize=6, markerfacecolor='black', label='collocation (x_i, y_i)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Mapping Normal → Gamma via Polynomial Interpolation')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main_demo()
