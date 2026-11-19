"""
svmlib.py -- squared-hinge (L2-loss) kernel SVM: primal and Lagrange dual, BOTH SOLVED WITH CVXPY.

Primal (K + eps*I = L L'):
    min_{u,b,xi}  1/2 |u|^2 + C sum xi_i^2     s.t.  y_i (L_i u + b) >= 1 - xi_i
Dual (derived by hand in the report), H = (y y') o K_eps + I/(2C):
    max_{alpha}   1'alpha - 1/2 alpha' H alpha   s.t.  alpha >= 0,  y'alpha = 0
      (written below with H alpha = L'(y o alpha) so that CVXPY sees a sum of squares)
Recovery:   c = alpha o y,   f(x) = sum_i c_i k(x_i, x) + b,   xi = alpha/(2C).
"""
import time
import numpy as np
import scipy.linalg as sl
import cvxpy as cp          # required: this module does all solving through CVXPY

EPS = 1e-6                  # jitter: K_eps = K + EPS*I is positive definite; used in BOTH problems
SOLVER = "CLARABEL"         # CVXPY solver name; set from main.py (--solver)


# ---------------------------------------------------------------- kernels and factor
def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


def kernel(A, B, kind="rbf", gamma=0.02):
    if kind == "linear":
        return A @ B.T
    return np.exp(-gamma * sqdist(A, B))


def factor(K):
    """K_eps = K + EPS I and its Cholesky factor L (K_eps = L L')."""
    Keps = K + EPS * np.eye(len(K))
    return Keps, np.linalg.cholesky(Keps)


# ---------------------------------------------------------------- the two problems
def solve_primal(L, y, C):
    n = len(y)
    u = cp.Variable(n); b = cp.Variable(); xi = cp.Variable(n)
    margin = cp.multiply(y, L @ u + b) >= 1 - xi
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u) + C * cp.sum_squares(xi)), [margin])
    prob.solve(solver=SOLVER)
    prob.is_dcp(dpp=True)
    return u.value, float(b.value), float(prob.value), prob.status


def dual_problem(L, C):
    """Dual with y as a CVXPY Parameter, so one problem object serves all ten digits (compiled once)."""
    n = L.shape[0]
    a = cp.Variable(n)
    yp = cp.Parameter(n)
    obj = cp.sum(a) - 0.5 * cp.sum_squares(L.T @ cp.multiply(yp, a)) - cp.sum_squares(a) / (4 * C)
    return cp.Problem(cp.Maximize(obj), [a >= 0, yp @ a == 0]), a, yp


def solve_dual(prob, a, yp, y):
    yp.value = y
    prob.solve(solver=SOLVER)
    prob.is_dcp(dpp=True)
    return np.maximum(a.value, 0.0), float(prob.value), prob.status


# ---------------------------------------------------------------- recovery from the dual
def recover(Keps, y, alpha, C):
    """Kernel coefficients c = alpha o y and intercept b from complementary slackness (alpha_i > 0)."""
    c = alpha * y
    u_f = Keps @ c                                  # = L u, the scores without intercept
    sv = alpha > 1e-8 * max(alpha.max(), 1e-300)
    b = np.mean(y[sv] * (1 - alpha[sv] / (2 * C)) - u_f[sv]) if sv.any() else 0.0
    return c, b


# ---------------------------------------------------------------- model selection: OvR, dual only
def fit_ovr_dual(Keps, L, ylab, C):
    """10 one-versus-rest duals with CVXPY (one compiled problem, y swapped per digit)."""
    prob, a, yp = dual_problem(L, C)
    cs, bs = [], []
    for k in range(10):
        y = np.where(ylab == k, 1.0, -1.0)
        alpha, _, status = solve_dual(prob, a, yp, y)
        c, b = recover(Keps, y, alpha, C)
        cs.append(c); bs.append(b)
    return np.array(cs), np.array(bs)


# ---------------------------------------------------------------- final: one digit, primal AND dual
def solve_binary(Keps, L, y, C):
    """Solve primal and dual of one binary problem with CVXPY; return everything needed for the checks."""
    out = {}
    t = time.perf_counter(); u, b_p, pv, ps = solve_primal(L, y, C); out["tp"] = time.perf_counter() - t
    prob, a, yp = dual_problem(L, C)
    t = time.perf_counter(); alpha, dv, ds = solve_dual(prob, a, yp, y); out["td"] = time.perf_counter() - t
    c_d, b_d = recover(Keps, y, alpha, C)
    c_p = sl.solve_triangular(L.T, u, lower=False)  # c = L^{-T} u
    out.update(pv=pv, dv=dv, gap=abs(pv - dv), alpha=alpha, c_p=c_p, b_p=b_p, c_d=c_d, b_d=b_d,
               status=(ps, ds))
    return out