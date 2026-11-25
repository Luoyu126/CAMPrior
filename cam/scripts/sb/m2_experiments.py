"""Numerical experiments of stage M2 (docs/20261125/math_route2.md sections 7-8): time discretisation of the
tau-leaping sampler and its step rule, temperature annealing of the alternation, and the sizes of the negative controls.
Run: python cam/scripts/sb/m2_experiments.py > docs/20261125/m2_experiments.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import dasbs_exact as de   # noqa: E402


def setup(n, A, tau, ref, lam=2.0):
    X = de.states(n); mu = np.full(2 ** n, 1.0 / 2 ** n); nu = de.boltzmann(X, A, lam, tau)
    return X, mu, nu, de.ExactSB(mu, nu, ref, n)


def discretisation():
    print('== 1. tau-leaping (DASBS Alg. 1, first-order, controller frozen at the left end) with the EXACT controller')
    print('graph n p tau schedule | TV(source, nu) | TV at 25 / 50 / 100 / 200 / 400 steps | max flip prob at 50 steps')
    for seed, (n, p) in enumerate([(8, 0.3), (8, 0.5), (7, 0.4)]):
        rng = np.random.default_rng(seed)
        A = de.random_graph(n, p, rng)
        for tau in (1.0, 0.5, 0.3):
            for ref in (de.Reference('constant', 1.5), de.Reference('loglinear', 1.0, 0.5)):
                X, mu, nu, sb = setup(n, A, tau, ref)
                row = []
                for steps in (25, 50, 100, 200, 400):
                    pend, worst = de.tau_leaping_marginal(mu, sb.controller, ref, sb.nb, steps)
                    row.append(de.tv(pend, nu))
                    if steps == 50:
                        w50 = worst
                mono = all(a > b for a, b in zip(row, row[1:]))
                print(f'{seed} {n} {p} {tau} {ref.schedule:9s} | {de.tv(mu, nu):.3f} | '
                      + ' / '.join(f'{v:.2e}' for v in row) + f' | {w50:.3f} | monotone={mono}')


def flip_rule():
    print('\n== 4. step rule of tau-leaping: linear (DASBS Prop. B.7, clipped) vs exp (1 - exp(-rate h)), exact controller')
    print('graph tau schedule | linear TV at 10 / 25 / 50 / 100 | exp TV at 10 / 25 / 50 / 100 | max linear flip prob at 10 / 50')
    for seed, (n, p) in enumerate([(8, 0.3), (8, 0.5), (7, 0.4)]):
        rng = np.random.default_rng(seed)
        A = de.random_graph(n, p, rng)
        for tau in (1.0, 0.5, 0.3):
            for ref in (de.Reference('constant', 1.5), de.Reference('loglinear', 1.0, 0.5)):
                X, mu, nu, sb = setup(n, A, tau, ref)
                lin, ex, worst = [], [], []
                for steps in (10, 25, 50, 100):
                    pl, w = de.tau_leaping_marginal(mu, sb.controller, ref, sb.nb, steps, 'linear')
                    pe, _ = de.tau_leaping_marginal(mu, sb.controller, ref, sb.nb, steps, 'exp')
                    lin.append(de.tv(pl, nu)); ex.append(de.tv(pe, nu)); worst.append(w)
                print(f'{seed} {tau} {ref.schedule:9s} | ' + ' / '.join(f'{v:.2e}' for v in lin) + ' | '
                      + ' / '.join(f'{v:.2e}' for v in ex) + f' | {worst[0]:.2f} / {worst[2]:.2f}')


def time_grid():
    print('\n== 5. time grid of tau-leaping (linear rule, exact controller): uniform vs t = 1 - (1 - s)^2 (denser near t = 1)')
    print('graph tau schedule | uniform TV at 10 / 25 / 50 / 100 | dense-end TV at 10 / 25 / 50 / 100')
    for seed, (n, p) in enumerate([(8, 0.3), (8, 0.5), (7, 0.4)]):
        rng = np.random.default_rng(seed)
        A = de.random_graph(n, p, rng)
        for tau in (1.0, 0.3):
            for ref in (de.Reference('constant', 1.5), de.Reference('loglinear', 1.0, 0.5)):
                X, mu, nu, sb = setup(n, A, tau, ref)
                uni, den = [], []
                for steps in (10, 25, 50, 100):
                    s_ = np.linspace(0, 1, steps + 1)
                    pu, _ = de.tau_leaping_marginal(mu, sb.controller, ref, sb.nb, steps)
                    pd, _ = de.tau_leaping_marginal(mu, sb.controller, ref, sb.nb, steps, grid=1 - (1 - s_) ** 2)
                    uni.append(de.tv(pu, nu)); den.append(de.tv(pd, nu))
                print(f'{seed} {tau} {ref.schedule:9s} | ' + ' / '.join(f'{v:.2e}' for v in uni) + ' | '
                      + ' / '.join(f'{v:.2e}' for v in den))


def annealing():
    print('\n== 2. temperature annealing of the exact alternation (controller = ctrl-AM fixed point, corrector = corr-DM)')
    rng = np.random.default_rng(3)
    n = 8; A = de.random_graph(n, 0.4, rng)
    X = de.states(n); mu = np.full(2 ** n, 1.0 / 2 ** n); nb = de.neighbours(n)
    ref = de.Reference('loglinear', 1.0, 0.5)
    taus = np.geomspace(2.0, 0.3, 8)

    def run(nu, log_phihat1, max_stages=2000, tol=1e-4):
        for k in range(1, max_stages + 1):
            P01, _ = de.forward_half_bridge(mu, nu, log_phihat1, ref, X)
            if de.tv(P01.sum(0), nu) < tol:
                return k, log_phihat1
            R = de.corr_dm_map(P01, ref, X, nb)
            log_phihat1, _ = de.ratios_to_log_potential(R, nb)
        return None, log_phihat1

    print('tau | stages to TV < 1e-4: cold start (all-one corrector) / warm start (previous tau) | '
          'TV if the corrector is NOT refitted after the change')
    warm = np.zeros(2 ** n)
    for tau in taus:
        nu = de.boltzmann(X, A, 2.0, tau)
        cold_k, _ = run(nu, np.zeros(2 ** n))
        P_stale, _ = de.forward_half_bridge(mu, nu, warm, ref, X)
        stale = de.tv(P_stale.sum(0), nu)
        warm_k, warm = run(nu, warm)
        print(f'{tau:.3f} | {cold_k} / {warm_k} | {stale:.2e}')

    print('\ncontinuous annealing: ONE alternation per temperature, 200 temperatures from 2.0 to 0.3, then fixed tau = 0.3')
    taus = np.geomspace(2.0, 0.3, 200)
    lphh = np.zeros(2 ** n); trace = []
    for j, tau in enumerate(taus):
        nu = de.boltzmann(X, A, 2.0, tau)
        P01, _ = de.forward_half_bridge(mu, nu, lphh, ref, X)
        trace.append(de.tv(P01.sum(0), nu))
        lphh, _ = de.ratios_to_log_potential(de.corr_dm_map(P01, ref, X, nb), nb)
    extra = []
    for k in range(50):
        P01, _ = de.forward_half_bridge(mu, nu, lphh, ref, X)
        extra.append(de.tv(P01.sum(0), nu))
        lphh, _ = de.ratios_to_log_potential(de.corr_dm_map(P01, ref, X, nb), nb)
    print('TV to the current target at temperature index 0/50/100/150/199:',
          ' '.join(f'{trace[i]:.2e}' for i in (0, 50, 100, 150, 199)))
    print('after the schedule ends, extra alternations 1/5/10/20/50:', ' '.join(f'{extra[i]:.2e}' for i in (0, 4, 9, 19, 49)))


def negatives():
    print('\n== 3. sizes of the negative controls (n = 3 graph of tests/test_dasbs_exact.py, tau = 0.7, constant gamma = 1.5)')
    rng = np.random.default_rng(0)
    n = 3; A = de.random_graph(n, 0.5, rng)
    ref = de.Reference('constant', 1.5)
    X, mu, nu, sb = setup(n, A, 0.7, ref)
    P_nc, _ = de.forward_half_bridge(mu, nu, np.zeros(2 ** n), ref, X)
    hist = de.ipf_alternation(mu, nu, ref, n, stages=400, log_ratio_corrector=True)
    print(f'no corrector (non-memoryless): TV {de.tv(P_nc.sum(0), nu):.3e}')
    print(f'corrector regresses log-ratio: TV {de.tv(hist[-1], nu):.3e}')
    print(f'right alternation, 400 stages: TV {de.tv(de.ipf_alternation(mu, nu, ref, n, stages=400)[-1], nu):.3e}')


if __name__ == '__main__':
    discretisation()
    annealing()
    flip_rule()
    time_grid()
    negatives()
