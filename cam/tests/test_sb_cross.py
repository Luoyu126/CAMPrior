"""Stage M3 of docs/20261125/plan.md: route 2's exact SB (utils/dasbs_exact.py, DASBS conventions) with the target
nu = pi1 on a label set equals route 1's Sinkhorn coupling (utils/sb_coupling.py, DIFUSCO conventions).

Correspondence (math_route2.md §9):
    route 2 time t in [0, 1], source mu = uniform at t = 0, target at t = 1
    route 1 index k, noise end z at k = T, data end y at k = 0
    route 1 signal between the data end and x_k  <->  exp(-Gamma(t, 1)) in route 2   (so a_T <-> exp(-Gamma(0, 1)))
    route 1 potential g_m                         <->  phi_1(s_m) up to a constant
Acceptance: couplings and the marginals at every time agree, TV < 1e-8; here also the label posteriors.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import dasbs_exact as de    # noqa: E402
from utils import sb_coupling as sb    # noqa: E402

TOL = 1e-8


def instances(count=12):
    out = []
    for seed in range(count):
        rng = np.random.default_rng(100 + seed)
        n = 5 + seed % 4
        M = 2 + seed % 5
        idx = rng.choice(2 ** n, size=M, replace=False)
        pi1 = rng.uniform(0.2, 1.0, M); pi1 /= pi1.sum()
        out.append((n, idx, pi1))
    return out


@pytest.mark.parametrize('sched', [('constant', 1.2, None), ('loglinear', 0.8, 0.5)])
def test_route2_exact_equals_route1_sinkhorn(sched):
    name, gam, alpha = sched
    ref = de.Reference(name, gam) if name == 'constant' else de.Reference(name, gam, alpha)
    worst = dict(coupling=0.0, marginal=0.0, posterior=0.0, potential=0.0)
    for n, idx, pi1 in instances():
        X = de.states(n); S = len(X)
        mu = np.full(S, 1.0 / S)
        nu = np.zeros(S); nu[idx] = pi1
        sb2 = de.ExactSB(mu, nu, ref, n)
        assert sb2.marginal_error < 1e-12
        assert np.abs(np.delete(sb2.joint, idx, axis=1)).max() == 0        # no mass off the labels
        J2 = sb2.joint[:, idx]                                              # (z, m)

        a_T = ref.signal(0.0, 1.0)
        Xt = torch.from_numpy(X); labels = Xt[idx]
        log_g, err, _ = sb.sinkhorn(labels, torch.from_numpy(pi1), a_T, Xt, tol=1e-14)
        assert err < 1e-13
        J1 = (sb.coupling_conditional(Xt, labels, log_g, a_T) / S).numpy()
        worst['coupling'] = max(worst['coupling'], de.tv(J1, J2))

        # g ∝ phi_1 on the labels
        lp = sb2.log_phi1[idx]
        d = log_g.numpy() - lp
        worst['potential'] = max(worst['potential'], np.abs(d - d.mean()).max())

        for t in (0.0, 0.2, 0.5, 0.8, 0.95, 1.0):
            a_k = ref.signal(t, 1.0)
            # route 1 marginal at the index with signal a_k: sum_{z,m} pi*(z, m) R(x | s_m, z)
            Ky = de.channel(X[idx], X, a_k)                                 # (m, x) K_{a_k}(x | s_m)
            if a_k > a_T:
                Kz = de.channel(X, X, a_T / a_k)                            # (x, z) K_{a_T/a_k}(z | x)
                Kyz = de.channel(X[idx], X, a_T)                            # (m, z)
                P1 = np.einsum('zm,mx,xz->x', J1 / Kyz.T, Ky, Kz)
            else:                                                           # t = 0: the noise end itself
                P1 = J1.sum(1)
            worst['marginal'] = max(worst['marginal'], de.tv(P1, sb2.marginal(t)))
            if t < 1.0:
                post1 = sb.label_log_posterior(Xt, labels, log_g, a_k).exp().numpy()          # (x, m)
                Kt1 = ref.kernel(X, t, 1.0)[:, idx]                                            # (x, m)
                post2 = Kt1 * np.exp(sb2.log_phi1[idx])[None, :] / sb2.phi(t)[:, None]
                worst['posterior'] = max(worst['posterior'], np.abs(post1 - post2).max())
    print(sched, {k: f'{v:.1e}' for k, v in worst.items()})
    assert worst['coupling'] < TOL and worst['marginal'] < TOL
    assert worst['posterior'] < TOL and worst['potential'] < 1e-8
