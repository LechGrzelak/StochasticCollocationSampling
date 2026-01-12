This code implements the core components of the Stochastic Collocation Monte Carlo (SCMC) method. It is organized into modular functions:

build_moment_matrix(moment_fn, N):        Constructs the (N+1)x(N+1) Hankel moment matrix M using a user-provided moment function m(k) = E[X^k].
gauss_quadrature_from_moment_matrix(M):   Computes recurrence coefficients (alpha, beta), builds the Jacobi matrix, and returns collocation nodes (Gauss points) and weights via eigen-decomposition.
lagrange_polynomial_values(y_i, x_i, X):  Evaluates the Lagrange interpolant at arbitrary points X, enabling mapping from a cheap base distribution to an expensive target distribution.

These components allow efficient sampling from computationally expensive distributions by combining a small number of exact evaluations with many cheap samples, following the SCMC approach.
Reference:

**Grzelak, L. A.; Witteveen, J. A. S.; Suárez-Taboada, M.; Oosterlee, C. W.
“The Stochastic Collocation Monte Carlo sampler: highly efficient sampling from ‘expensive’ distributions.” Quantitative Finance, 19(2):339–356, 2019. DOI: https://doi.org/10.1080/14697688.2018.1459807.**
