"""
pid_emergence.py — the three information-decomposition emergence tests of
Riedl & Rosas (Section 2.2) applied to a small discrete dynamical system.

What it computes
----------------
Given a lagged joint distribution p(s_t, s_{t+l}) over the microstates of an
n-part system (either exact, from a Markov transition matrix, or estimated by
plug-in counts from a trajectory), it computes:

  1. Emergence capacity  — median over part-pairs (i, j) of the bivariate
     dynamical synergy Syn(X_i,t, X_j,t ; (X_i,t+l, X_j,t+l)).
  2. Unique-information criterion, first order (k = 1) — whether a macro
     V_t = f(s_t) predicts the micro future s_{t+l} better than any single part:
     Un1 = I(V_t ; s_{t+l}) - max_i I(X_i,t ; s_{t+l}).
  3. Practical criterion — Psi_l(V) = I(V_t ; V_{t+l}) - sum_i I(X_i,t ; V_{t+l})
     (the Ψ of Rosas et al. 2020; Eq. 4 of the paper).

What it assumes about inputs
--------------------------------------------------
  * Microstates are discrete and enumerable: `states` is an S x n array whose
    row s is the value of each of the n parts in microstate s (any discrete
    alphabet; ±1 for the Ising toy).
  * `joint` is an S x S array with joint[s, s'] = p(s_t = s, s_{t+l} = s'),
    summing to 1.  Exact or estimated — every test takes the same object.
  * Redundancy is MMI (min of marginal MIs), Williams–Beer I_min, or BROJA
    (Bertschinger et al. 2014; binary sources only, see `broja_redundancy`).
    Three measures are implemented so that a second measure can be reported
    (Table S1).  On this system MMI and I_min coincide; BROJA does not.
  * All information quantities are in bits.
"""
from itertools import combinations

import numpy as np

LN2 = np.log(2.0)


# ----------------------------------------------------------------------------
# Information-theoretic primitives
# ----------------------------------------------------------------------------
def entropy(p):
    """Shannon entropy in bits of an (unnormalised) probability array."""
    p = np.asarray(p, dtype=float).ravel()
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / LN2)


def mutual_information(pab):
    """I(A;B) in bits from a 2-D joint array p[a, b]."""
    pab = np.asarray(pab, dtype=float)
    pab = pab / pab.sum()
    return entropy(pab.sum(1)) + entropy(pab.sum(0)) - entropy(pab)


def project(joint, row_labels, col_labels):
    """Marginalise joint[s, s'] onto (f(s), g(s')) given integer label arrays.

    Returns a (max(f)+1) x (max(g)+1) array.
    """
    S = joint.shape[0]
    R = np.zeros((row_labels.max() + 1, S))
    R[row_labels, np.arange(S)] = 1.0
    C = np.zeros((S, col_labels.max() + 1))
    C[np.arange(S), col_labels] = 1.0
    return R @ joint @ C


def combine_labels(*label_arrays):
    """Map a tuple of integer label arrays to a single dense integer label."""
    stacked = np.stack(label_arrays, axis=1)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return inv.ravel()


# ----------------------------------------------------------------------------
# Bivariate PID
# ----------------------------------------------------------------------------
def imin_redundancy(p):
    """Williams & Beer (2010) I_min redundancy for a 3-D joint p[x1, x2, t]."""
    p = p / p.sum()
    pt = p.sum(axis=(0, 1))
    marg = [p.sum(1), p.sum(0)]  # p(x1, t), p(x2, t)
    red = 0.0
    for t in range(p.shape[2]):
        if pt[t] <= 0:
            continue
        spec = []
        for pst in marg:
            ps = pst.sum(1)
            col = pst[:, t]
            m = col > 0
            p_s_given_t = col[m] / pt[t]
            p_t_given_s = col[m] / ps[m]
            spec.append(float((p_s_given_t * np.log(p_t_given_s / pt[t])).sum() / LN2))
        red += pt[t] * min(spec)
    return red


def broja_redundancy(p, tol=1e-12, max_sweeps=500):
    """BROJA redundancy (Bertschinger, Rauh, Olbrich, Jost & Ay 2014) for a 3-D joint
    p[x1, x2, t] with BINARY sources x1, x2 and any discrete target t.

    BROJA defines synergy as  Syn = I_P(X1,X2;T) - min_{Q in Δ_P} I_Q(X1,X2;T),
    where Δ_P is the set of joints with the same pairwise marginals p(x1,t) and
    p(x2,t) as P.  Redundancy then follows from Eq. (1): Red = I1 + I2 - I12 + Syn.

    Assumes binary sources: then for each value t the 2x2 slice Q[:, :, t] has
    fixed row and column sums, so Δ_P is a box with one free scalar per t and the
    convex objective is minimised by cyclic coordinate descent with an exact
    (golden-section) line search.  No dependency beyond numpy.  Verified against
    scipy L-BFGS-B on this system (agreement <= 1e-8 bits) and against the
    closed-form AND / XOR / UNQ / RDN gates of Bertschinger et al. (Table 1) in
    `test_invariants.py`.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    if p.shape[0] != 2 or p.shape[1] != 2:
        raise ValueError("broja_redundancy: sources must be binary; use a Table S3 package otherwise")
    r, c, pt = p.sum(1), p.sum(0), p.sum((0, 1))  # p(x1,t), p(x2,t), p(t)
    nt = p.shape[2]
    lo = np.maximum(0.0, r[0] + c[0] - pt)  # bounds on q_t = Q[0, 0, t]
    hi = np.minimum(r[0], c[0])

    def joint_mi(q):  # I_Q(X1,X2;T) for the member of Δ_P with Q[0,0,:] = q
        Q = np.empty((2, 2, nt))
        Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1] = q, r[0] - q, c[0] - q, q - r[0] - c[0] + pt
        Q = np.clip(Q, 0.0, None)
        return entropy(Q.sum(2)) + entropy(pt) - entropy(Q)

    q = np.clip((lo + hi) / 2, lo, hi)
    f = joint_mi(q)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(max_sweeps):
        f_old = f
        for t in range(nt):
            a, b = lo[t], hi[t]
            if b - a < 1e-15:
                continue
            x1, x2 = b - g * (b - a), a + g * (b - a)
            qa = q.copy()
            qa[t] = x1; f1 = joint_mi(qa)
            qa[t] = x2; f2 = joint_mi(qa)
            while b - a > 1e-13:
                if f1 < f2:
                    b, x2, f2 = x2, x1, f1
                    x1 = b - g * (b - a); qa[t] = x1; f1 = joint_mi(qa)
                else:
                    a, x1, f1 = x1, x2, f2
                    x2 = a + g * (b - a); qa[t] = x2; f2 = joint_mi(qa)
            q[t] = (a + b) / 2
            f = joint_mi(q)
        if f_old - f < tol:
            break
    I1 = mutual_information(r)
    I2 = mutual_information(c)
    I12 = mutual_information(p.reshape(-1, nt))
    syn = I12 - f
    return I1 + I2 - I12 + syn


def bivariate_pid(p, redundancy="mmi"):
    """Two-source PID (Eq. 1 of the paper) from a 3-D joint p[x1, x2, t].
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    I1 = mutual_information(p.sum(1))
    I2 = mutual_information(p.sum(0))
    I12 = mutual_information(p.reshape(-1, p.shape[2]))
    if redundancy == "mmi":
        red = min(I1, I2)
    elif redundancy == "imin":
        red = imin_redundancy(p)
    elif redundancy == "broja":
        red = broja_redundancy(p)
    else:
        raise ValueError(redundancy)
    return {
        "I1": I1, "I2": I2, "I12": I12,
        "red": red, "un1": I1 - red, "un2": I2 - red,
        "syn": I12 - I1 - I2 + red,
    }


# ----------------------------------------------------------------------------
# The three emergence tests.  Every test takes (states, joint, ...).
# ----------------------------------------------------------------------------
def _labels(states, i):
    """Dense integer labels of part i's alphabet (works for any discrete alphabet)."""
    return np.unique(states[:, i], return_inverse=True)[1].ravel()


def emergence_capacity(states, joint, redundancy="mmi"):
    """Test 1.  Pairwise dynamical synergy, median over all unordered pairs.

    For each pair (i, j): sources X_i,t and X_j,t; target the joint future
    (X_i,t+l, X_j,t+l).  
    """
    S, n = states.shape
    si, sj = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    out = {}
    for i, j in combinations(range(n), 2):
        x1, x2 = _labels(states, i), _labels(states, j)
        t = combine_labels(x1, x2)  # label of the joint future (X_i,t+l, X_j,t+l)
        p = np.zeros((x1.max() + 1, x2.max() + 1, t.max() + 1))
        np.add.at(p, (x1[si], x2[si], t[sj]), joint)
        out[(i, j)] = bivariate_pid(p, redundancy)
    syn = np.array([v["syn"] for v in out.values()])
    red = np.array([v["red"] for v in out.values()])
    return {"pairs": out, "syn_median": float(np.median(syn)),
            "red_median": float(np.median(red))}


def unique_information_criterion(states, joint, macro):
    """Test 2, first order (k = 1).

    Un1 = I(V_t ; s_{t+l}) - max_i I(X_i,t ; s_{t+l}).
    This is the bivariate MMI unique information of V against the single most
    predictive part.  
    """
    S, n = states.shape
    idx = np.arange(S)
    V = combine_labels(macro(states))
    I_V = mutual_information(project(joint, V, idx))
    I_parts = [mutual_information(project(joint, _labels(states, i), idx)) for i in range(n)]
    I_X = mutual_information(joint)
    I_VX = mutual_information(project(joint, combine_labels(V, idx), idx))
    return {"I_V": I_V, "I_parts": I_parts, "un1": I_V - max(I_parts),
            "full_order_conditional": I_VX - I_X}


def practical_criterion(states, joint, macro):
    """Test 3.  Psi_l(V) = I(V_t ; V_{t+l}) - sum_i I(X_i,t ; V_{t+l})  (Eq. 4; Ψ of Rosas et al. 2020)."""
    S, n = states.shape
    V = combine_labels(macro(states))
    I_VV = mutual_information(project(joint, V, V))
    I_parts = [mutual_information(project(joint, _labels(states, i), V)) for i in range(n)]
    return {"I_VV": I_VV, "I_parts": I_parts, "psi": I_VV - sum(I_parts)}


def run_all_tests(states, joint, macro, redundancy="mmi"):
    """Convenience wrapper returning the three scalar test statistics."""
    cap = emergence_capacity(states, joint, redundancy)
    uic = unique_information_criterion(states, joint, macro)
    prc = practical_criterion(states, joint, macro)
    return {"capacity_syn": cap["syn_median"], "capacity_red": cap["red_median"],
            "un1": uic["un1"], "psi": prc["psi"]}


# ----------------------------------------------------------------------------
# Toy system: kinetic Ising ring with synchronous Glauber dynamics
# ----------------------------------------------------------------------------
def ising_states(n):
    """All 2^n spin configurations as an (2^n, n) array of ±1."""
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    return 2 * bits - 1


def ising_transition_matrix(n, beta, J=1.0, h=0.0):
    """Synchronous Glauber dynamics on a ring of n ±1 spins.

    P[s, s'] = prod_i  sigma( 2 * beta * s'_i * (J (s_{i-1} + s_{i+1}) + h) ).
    """
    states = ising_states(n)
    local = J * (np.roll(states, 1, axis=1) + np.roll(states, -1, axis=1)) + h  # (S, n)
    # prob that spin i takes value s'_i given current state s
    logits = 2.0 * beta * states[None, :, :] * local[:, None, :]  # (S, S', n)
    P = np.prod(1.0 / (1.0 + np.exp(-logits)), axis=2)
    return states, P


def stationary_distribution(P, tol=1e-14, max_iter=100000):
    """Stationary distribution by power iteration (chain is ergodic for beta < inf)."""
    S = P.shape[0]
    pi = np.full(S, 1.0 / S)
    for _ in range(max_iter):
        new = pi @ P
        if np.abs(new - pi).max() < tol:
            return new / new.sum()
        pi = new
    return pi / pi.sum()


def exact_lagged_joint(P, lag=1):
    """Exact p(s_t, s_{t+l}) = pi(s) [P^l](s, s') under stationarity."""
    pi = stationary_distribution(P)
    Pl = np.linalg.matrix_power(P, lag)
    return pi[:, None] * Pl


def magnetisation(states):
    """Macro variable V = sum of spins."""
    return states.sum(1)


# ----------------------------------------------------------------------------
# Estimation layer: sampled trajectories, plug-in joint, permutation null
# ----------------------------------------------------------------------------
def simulate(P, T, seed, burn=1000):
    """Sample a trajectory of state indices of length T after `burn` steps."""
    rng = np.random.default_rng(seed)
    cum = P.cumsum(1)
    s = int(rng.integers(P.shape[0]))
    traj = np.empty(T + burn, dtype=int)
    u = rng.random(T + burn)
    for t in range(T + burn):
        traj[t] = s
        s = int(np.searchsorted(cum[s], u[t]))
    return traj[burn:]


def empirical_joint(a, b, S):
    """Plug-in joint from paired index arrays."""
    J = np.zeros((S, S))
    np.add.at(J, (a, b), 1.0)
    return J / J.sum()


def sampled_tests(states, P, T, seed, lag=1, n_null=50, redundancy="mmi", macro=magnetisation):
    """Raw plug-in estimates, permutation-null mean, and bias-corrected (BC) values."""
    S = P.shape[0]
    traj = simulate(P, T, seed)
    a, b = traj[:-lag], traj[lag:]
    raw = run_all_tests(states, empirical_joint(a, b, S), macro, redundancy)
    rng = np.random.default_rng(seed + 1)
    nulls = []
    for _ in range(n_null):
        nulls.append(run_all_tests(states, empirical_joint(a, rng.permutation(b), S), macro, redundancy))
    null_mean = {k: float(np.mean([d[k] for d in nulls])) for k in raw}
    null_sd = {k: float(np.std([d[k] for d in nulls])) for k in raw}
    bc = {k: raw[k] - null_mean[k] for k in raw}
    return {"raw": raw, "null_mean": null_mean, "null_sd": null_sd, "bc": bc, "T": T, "seed": seed}
