# TSP-500: model-sample labels vs LKH-3 labels, random vs soft (DIFUSCO)

Extends the RB-small finding to a non-QUBO problem: does a trained solver's own samples make more learnable
labels than a strong classical solver's, and do soft labels help?

The four groups (two label sources × two ways of using the label set):

| | random (a random one of the best tours, "最大解") | soft (posterior mean over the set) |
|---|---|---|
| LKH-3 tours | `lkh_random` | `lkh_soft` |
| CAM samples | `cam_random` | `cam_soft` |

## Data

All files are in `/usr0/home/shengyuf/yunyi/data/tsp500_sl/` and are not in git.

- **Training instances:** 12,800 instances of 500 points, uniform on [0,1]², generated with `gen_instances.py --seed 1234`. DIFUSCO used 128k; we use fewer for compute.
- **LKH-3 labels:**
  - Script: `cam/scripts/tsp/lkh_labels.py`, 32 seeds per instance, RUNS=1, MAX_TRIALS=1. Coordinates are scaled ×1e6 and rounded.
  - We keep the distinct tours.
  - On 16 test instances, the best of the 32 tours is 0.22% shorter than the file's reference tour, and the 8th best is 0.08% shorter.
- **CAM labels:**
  - The released CAM TSP-500 checkpoint, run with the CAM paper's settings: 10 steps, 16 samples, greedy merge, 1000 2-opt iterations.
  - Script: `cam/scripts/tsp/dump_tours.py`.
- **Matching (`match_tour_sets.py --max_keep 8`):** for each instance, both sources keep their M shortest tours, M = min(8, number of tours from each source). Both sources therefore have the same number of labels per instance.
- **How a tour becomes a label (`cam/utils/tour_sets.py`):** each tour marks its edges in both directions on the sparse kNN graph. This marking is used in every group.

## Training (`launch.sh`)

The network and optimizer follow DIFUSCO TSP-500 (Sun & Yang 2023, App. E):
- AnGNN, 12 layers, width 256, sparse k=50;
- AdamW, lr 2e-4 with cosine decay, weight decay 1e-4.

The diffusion settings follow our MIS SOP: 50 cosine diffusion steps, and 50 steps at inference.

Differences from DIFUSCO, for compute:

| Setting | Ours | DIFUSCO |
|---|---|---|
| Training instances | 12.8k | 128k |
| Initialization | from scratch | TSP-100 checkpoint |
| Batch | 16 (2 GPUs × 8) | 64 |
| Epochs | 50 | 50 |

Checkpoint selection: 16 validation instances, renoise, 4 samples, no 2-opt.

## Test (`test.sh`)

- Data: the 128 TSP-500 test instances (`tsp500_test_concorde.txt`).
- Sampling: 50 steps, 16 samples, greedy merge, best of 16.
- Two settings:
  - renoise without 2-opt;
  - renoise followed by 1000 2-opt iterations.
- Reported as the gap to the reference tours in the file.
- CAM's own result under its protocol: 16.916 (`runs/tsp_cam_eval/tsp500_2opt1000`, 127 instances; see the note below).

Note: the TSP dataset reads `lines[:num_samples]`, so with the default `-1` the last instance is silently dropped. We always pass `--test_samples 128`.

## Results

(pending)
