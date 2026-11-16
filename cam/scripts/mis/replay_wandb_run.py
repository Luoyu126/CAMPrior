"""Replay an existing wandb run into a new run, adding the positive MIS size at the original steps.

Used for the imported CAM-paper DiffUCO runs: their `eval/mis_size` had been appended after the
original history (steps > 30000, x-axis `epoch_idx`), so it did not line up with runs logged by the
current code, where `eval/mis_size` sits on the same step as `eval/object` (100 steps per epoch).

The new run gets the same name, config (minus wandb's own `_wandb` block), tags and notes, every
original history row at its original step (rows added later, i.e. those with `epoch_idx`, are
dropped), `eval/mis_size = -eval/object` on every validation row, and the summary keys given by
--summary_keys copied over. System metrics and wall-clock timestamps are not replayed.

Usage:
  python replay_wandb_run.py --src yunyi-organization/SLVUL/<id> --dir <wandb dir> \
      --summary_keys eval/mis_size_best test/mis_size test/mis_size_0514 test/time_per_graph_s
Prints the new run id. The source run is left untouched; delete it yourself after checking.
"""
import argparse

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='entity/project/run_id of the run to replay')
    p.add_argument('--dir', required=True, help='local directory for the new run\'s wandb files')
    p.add_argument('--summary_keys', nargs='*', default=[])
    p.add_argument('--note', default='Replayed from the imported run so that eval/mis_size sits on the same steps as eval/object.')
    args = p.parse_args()

    api = wandb.Api()
    src = api.run(args.src)
    entity, project, _ = args.src.split('/')
    config = {k: v for k, v in src.config.items() if k != '_wandb'}
    rows = [r for r in src.scan_history(page_size=1000) if r.get('epoch_idx') is None]
    rows.sort(key=lambda r: r['_step'])

    run = wandb.init(entity=entity, project=project, name=src.name, config=config,
                     tags=list(src.tags), notes=((src.notes or '') + '\n' + args.note).strip(),
                     dir=args.dir)
    n_eval = 0
    for r in rows:
        data = {k: v for k, v in r.items() if not k.startswith('_') and v is not None}
        if 'eval/object' in data:
            data['eval/mis_size'] = -data['eval/object']
            n_eval += 1
        if data:
            run.log(data, step=r['_step'])
    for k in args.summary_keys:
        if k in src.summary:
            run.summary[k] = src.summary[k]
    run.summary['replayed_from'] = src.id
    new_id = run.id
    run.finish()
    print(f'replayed {len(rows)} rows ({n_eval} validation points) from {src.id} into {new_id}')


if __name__ == '__main__':
    main()
