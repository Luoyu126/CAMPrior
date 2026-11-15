"""Fill the MIS graph cache (CAM_GRAPH_CACHE, see datasets/mis.py) in parallel (local addition,
see cam/UPSTREAM.md). Parsing a dense ER-[700-800] gpickle takes ~1-3 s, so supervised training,
which reads every training graph each epoch, needs the cache.

    source cam/scripts/env.sh
    cd cam && CAM_GRAPH_CACHE=$CAM_CACHE/graphs nice -n 10 $CAM_PYTHON scripts/mis/build_graph_cache.py \
        --workers 24 $CAM_ER700_DIR/er_train $CAM_ER700_DIR/er_valid $CAM_ER700_DIR/er_test
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import datasets.mis as M  # noqa: E402

_ds = None


def _init(path):
    global _ds
    _ds = M.Dataset(path)


def _load(i):
    _ds[i]
    return i


def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+')
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    assert os.environ.get('CAM_GRAPH_CACHE'), 'set CAM_GRAPH_CACHE'
    for path in args.paths:
        n = len(M.Dataset(path))
        t = time.time()
        with Pool(args.workers, initializer=_init, initargs=(path,)) as pool:
            for k, _ in enumerate(pool.imap_unordered(_load, range(n), chunksize=8), 1):
                if k % 1000 == 0 or k == n:
                    print(f'{path}: {k}/{n} graphs, {time.time() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
