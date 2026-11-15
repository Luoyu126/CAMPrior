"""Label MIS training graphs with KaMIS (redumis), as the DIFUSCO ER-[700-800] training set was.

Follows the DIMES / Boether et al. MIS benchmark framework (solvers/kamis.py) used by DIFUSCO
(data/README.md: `main.py solve kamis ... --time_limit 60`): the graph is written in METIS format
(1-based adjacency lists), redumis is called with --output and --time_limit, and the result is one
0/1 per line in <anno_dir>/<stem>_unweighted.result, the file the dataset loader reads.

    python cam/scripts/mis/label_kamis.py --graphs DIR --anno DIR --time_limit 60 --procs 40 \
        [--redumis PATH] [--tmp DIR] [--limit N] [--files_from LIST]
"""
import argparse
import glob
import os
import pickle
import subprocess
import time
from multiprocessing import Pool

REDUMIS = '/usr0/home/shengyuf/yunyi/tools/KaMIS/deploy/redumis'


def metis(G):
    lines = [f'{G.number_of_nodes()} {G.number_of_edges()} 0']
    for _, nbrs in sorted(G.adjacency()):
        lines.append(' '.join(str(v + 1) for v in sorted(nbrs)))
    return '\n'.join(lines) + '\n'


def label(job):
    path, anno, tmp, tl, redumis = job
    stem = os.path.basename(path)[:-len('.gpickle')]
    out = os.path.join(anno, f'{stem}_unweighted.result')
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return stem, None, 0.0
    with open(path, 'rb') as f:
        G = pickle.load(f)
    assert sorted(G.nodes) == list(range(G.number_of_nodes()))
    gfile = os.path.join(tmp, f'{stem}.graph')
    with open(gfile, 'w') as f:
        f.write(metis(G))
    part = out + '.part'
    t0 = time.monotonic()
    subprocess.run([redumis, gfile, '--output', part, '--time_limit', str(tl)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dt = time.monotonic() - t0
    x = [int(l) for l in open(part) if l.strip()]
    assert len(x) == G.number_of_nodes()
    assert not any(x[u] and x[v] for u, v in G.edges), f'{stem}: not independent'
    os.replace(part, out)
    os.remove(gfile)
    return stem, sum(x), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--anno', required=True)
    ap.add_argument('--time_limit', type=int, default=60)
    ap.add_argument('--procs', type=int, default=40)
    ap.add_argument('--redumis', default=REDUMIS)
    ap.add_argument('--tmp', default='/usr0/home/shengyuf/yunyi/.cache/kamis_tmp')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--files_from', default=None, help='text file with one gpickle path per line')
    a = ap.parse_args()
    os.makedirs(a.anno, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    if a.files_from:
        files = [l.strip() for l in open(a.files_from) if l.strip()]
    else:
        files = sorted(glob.glob(os.path.join(a.graphs, '*.gpickle')))
    files = files[:a.limit]
    jobs = [(p, a.anno, a.tmp, a.time_limit, a.redumis) for p in files]
    n_new, t0 = 0, time.time()
    with Pool(a.procs) as pool:
        for k, (stem, size, dt) in enumerate(pool.imap_unordered(label, jobs)):
            if size is not None:
                n_new += 1
                print(f'{stem} size {size} time {dt:.1f}', flush=True)
            if (k + 1) % 1000 == 0:
                print(f'# {k + 1}/{len(jobs)} done, {n_new} new, {(time.time() - t0) / 3600:.2f} h', flush=True)
    print(f'# finished {len(jobs)} graphs, {n_new} new', flush=True)


if __name__ == '__main__':
    main()
