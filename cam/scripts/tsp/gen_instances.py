"""Random uniform TSP instances in DIFUSCO's text format (local addition, UPSTREAM.md item 61).

Each line: x1 y1 x2 y2 ... output 1 2 ... n 1. The tour is a placeholder (identity); real labels come from
lkh_labels.py, which also writes the file with its best tour. Points are i.i.d. uniform on [0, 1]^2, as in
DIFUSCO's TSP-50/100/500/1000 training sets (Sun & Yang 2023, sec. 5.1).

    python cam/scripts/tsp/gen_instances.py OUT.txt --num 12800 --nodes 500 --seed 1234
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--num', type=int, required=True)
    ap.add_argument('--nodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tour = ' '.join(str(i + 1) for i in range(a.nodes)) + ' 1'
    with open(a.out, 'w') as f:
        for _ in range(a.num):
            p = rng.random((a.nodes, 2))
            f.write(' '.join(f'{v:.8f}' for v in p.reshape(-1)) + ' output ' + tour + '\n')


if __name__ == '__main__':
    main()
