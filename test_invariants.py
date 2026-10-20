"""Invariant and limiting-case checks for pid_emergence.py (run: pytest -q).

Only properties that are provable are asserted here; empirical regularities
(e.g. the location of the synergy peak) are reported in the notebook, not asserted.
"""
import numpy as np
import pytest

import pid_emergence as pe


def energy(states):
    return (states != np.roll(states, 1, axis=1)).sum(1)


BETAS = [0.0, 0.3, 1.0, 2.0]


@pytest.mark.parametrize("beta", BETAS)
def test_transition_matrix_is_stochastic(beta):
    _, P = pe.ising_transition_matrix(5, beta)
    assert np.allclose(P.sum(1), 1.0)


@pytest.mark.parametrize("beta", BETAS)
def test_joint_is_normalised_and_stationary(beta):
    _, P = pe.ising_transition_matrix(5, beta)
    J = pe.exact_lagged_joint(P)
    assert np.isclose(J.sum(), 1.0)
    assert np.allclose(J.sum(1), J.sum(0), atol=1e-10)  # stationarity: marginals of s_t and s_{t+1} agree


def test_independent_parts_give_zero_everywhere():
    """Limiting case: beta = 0 means every spin is an independent coin flip at every step."""
    st, P = pe.ising_transition_matrix(5, 0.0)
    J = pe.exact_lagged_joint(P)
    for macro in (pe.magnetisation, energy):
        r = pe.run_all_tests(st, J, macro)
        for k, v in r.items():
            assert abs(v) < 1e-9, (k, v)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("redundancy", ["mmi", "imin", "broja"])
def test_pid_atoms_nonnegative_and_sum_to_joint_mi(beta, redundancy):
    st, P = pe.ising_transition_matrix(5, beta)
    J = pe.exact_lagged_joint(P)
    cap = pe.emergence_capacity(st, J, redundancy)
    for pair, a in cap["pairs"].items():
        for atom in ("red", "un1", "un2", "syn"):
            assert a[atom] > -1e-10, (pair, atom, a[atom])
        assert np.isclose(a["red"] + a["un1"] + a["un2"] + a["syn"], a["I12"], atol=1e-10)


@pytest.mark.parametrize("beta", BETAS)
def test_full_order_unique_information_of_deterministic_macro_is_zero(beta):
    """Un(V_t; s_{t+l} | s_t) = 0 whenever V_t = f(s_t), for any non-negative PID."""
    st, P = pe.ising_transition_matrix(5, beta)
    J = pe.exact_lagged_joint(P)
    for macro in (pe.magnetisation, energy):
        u = pe.unique_information_criterion(st, J, macro)
        assert abs(u["full_order_conditional"]) < 1e-9


@pytest.mark.parametrize("beta", BETAS)
def test_flip_invariant_macro_has_zero_single_part_terms(beta):
    """With h = 0 the dynamics are symmetric under global spin flip, so a
    flip-invariant macro (energy) is independent of every single spin."""
    st, P = pe.ising_transition_matrix(5, beta)
    J = pe.exact_lagged_joint(P)
    p = pe.practical_criterion(st, J, energy)
    assert max(abs(x) for x in p["I_parts"]) < 1e-9


def test_sampled_estimate_approaches_exact():
    st, P = pe.ising_transition_matrix(5, 1.0)
    exact = pe.run_all_tests(st, pe.exact_lagged_joint(P), pe.magnetisation)
    est = pe.sampled_tests(st, P, T=50000, seed=20261008, n_null=10)
    for k in exact:
        assert abs(est["bc"][k] - exact[k]) < 0.02, (k, est["bc"][k], exact[k])


def _gate(fn, n_t=4):
    """Joint p[x1, x2, t] of a Boolean gate t = fn(x1, x2) with independent uniform inputs."""
    p = np.zeros((2, 2, n_t))
    for a in (0, 1):
        for b in (0, 1):
            p[a, b, fn(a, b)] += 0.25
    return p


@pytest.mark.parametrize("name, fn, expected", [
    # Bertschinger et al. (2014), Table 1: (red, un1, un2, syn) in bits
    ("AND", lambda a, b: a & b, (0.311278, 0.0, 0.0, 0.5)),
    ("XOR", lambda a, b: a ^ b, (0.0, 0.0, 0.0, 1.0)),
    ("UNQ", lambda a, b: 2 * a + b, (0.0, 1.0, 1.0, 0.0)),
])
def test_broja_matches_closed_form_gates(name, fn, expected):
    a = pe.bivariate_pid(_gate(fn), "broja")
    for atom, val in zip(("red", "un1", "un2", "syn"), expected):
        assert abs(a[atom] - val) < 1e-6, (name, atom, a[atom], val)


def test_broja_rdn_gate():
    """RDN: x1 = x2 = t is pure redundancy under BROJA."""
    p = np.zeros((2, 2, 2)); p[0, 0, 0] = p[1, 1, 1] = 0.5
    a = pe.bivariate_pid(p, "broja")
    assert abs(a["red"] - 1.0) < 1e-9 and abs(a["syn"]) < 1e-9


def test_broja_rejects_nonbinary_sources():
    with pytest.raises(ValueError):
        pe.broja_redundancy(np.full((3, 2, 2), 1 / 12))
