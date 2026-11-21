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

Test set: 128 TSP-500 instances. Reference tours average 16.5836. Each model is evaluated with one test seed. Paired differences are bootstrapped over instances, with 95% CIs (`summarize.py`).

### 50 steps, 16 samples, no 2-opt

| run | D3PM posterior | renoise |
|---|---|---|
| lkh_random | +13.32% | +12.08% |
| lkh_soft | +11.96% | **+10.98%** |
| cam_random | +14.86% | +13.11% |
| cam_soft | +15.24% | +12.90% |

For comparison, the released CAM model with its own protocol (10 steps, 16 samples, no 2-opt) gets +13.91%.

Paired differences, renoise:

| Comparison | Gap difference | 95% CI |
|---|---|---|
| soft − random, LKH labels | −1.11% | [−1.51, −0.68] |
| soft − random, CAM labels | −0.21% | [−0.70, +0.29] |
| CAM − LKH, random | +1.03% | [+0.57, +1.49] |
| CAM − LKH, soft | +1.93% | [+1.49, +2.36] |

Ten steps instead of 50 give almost the same results: lkh_random +12.20%, lkh_soft +11.08%; soft − random = −1.13% [−1.55, −0.70].

### With 1000 2-opt iterations (renoise, 16 samples)

| run | 50 steps | 10 steps (CAM's protocol) |
|---|---|---|
| lkh_random | +1.572% | +1.570% |
| lkh_soft | **+1.415%** | +1.482% |
| cam_random | +1.957% | |
| cam_soft | +1.982% | |
| CAM itself (UL, released checkpoint) | | +1.976% |

Reported numbers for reference (CAM paper, Table 2, same 10-step / 16-sample / 2-opt protocol): DIFUSCO trained on 128k LKH labels, +1.81%; CAM, +2.18%.

Paired differences:

| Comparison | Gap difference | 95% CI |
|---|---|---|
| lkh_soft (10 steps) − CAM | −0.494% | [−0.596, −0.392] |
| lkh_random (10 steps) − CAM | −0.407% | [−0.492, −0.322] |
| cam_random − CAM | −0.019% | [−0.105, +0.065] (the student matches its teacher) |
| soft − random, LKH labels, 50 steps | −0.157% | [−0.233, −0.081] |
| soft − random, CAM labels, 50 steps | +0.025% | [−0.050, +0.099] |
| CAM − LKH, random | +0.385% | [+0.304, +0.467] |
| CAM − LKH, soft | +0.567% | [+0.488, +0.648] |

Time per instance (one process): CAM 7.2 s. Ours with 10 steps 12.5 s, measured while the GPU was shared with MIS training, so it is not directly comparable.

Pending: decode-before-renoise (5 2-opt per step, no final 2-opt).
