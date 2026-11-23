# Vendored CAM code

- Source: https://github.com/Shengyu-Feng/CAM (official code of "Unsupervised Diffusion Solver
  for Combinatorial Optimization via Combinatorial Adjoint Matching", Feng, Suresh and Yang,
  ICML 2026).
- Upstream commit: `b8c6167add0b77b54c3013252f0c93a6c298cb9a` ("FM schedule", 2026-10-27).
- Vendored on 2026-11-06 from a read-only clone of that commit, all files except `.git/`.
- License: the upstream repository has no LICENSE file. Used with the author's permission.

## Local modifications

Everything else is byte-identical to the upstream commit (`diff -r <upstream clone> cam`
shows only the items below).

1. `cam.png` (README figure) not copied, so the image link in `README.md` is broken.
2. `main.py`: when `--do_train` and `--save_dir` already holds `*.pt`, upstream asks for
   confirmation with `input()`, which hangs unattended runs. It now exits at once with an error
   unless `CAM_ALLOW_OVERWRITE=1` is set.
3. `solvers/meta_solver.py`, `MetaSolver.evaluate`: also records the dataset index of each
   evaluated graph, and when a log file is given (test mode) writes
   `<log_file>.per_graph.csv` with columns `idx,file,obj,time` (objective = -MIS size, time in
   seconds, same values that upstream averages). Averages, validation and training are unchanged.
4. Added `scripts/env.sh`: paths to the existing conda env `~/anaconda3/envs/CAM` (used
   read-only), data/run dirs, and every cache/config/temp dir pointed under
   `/usr0/home/shengyuf/yunyi/.cache/cam`, wandb disabled unless `CAM_WANDB=1` (then online to entity
   `yunyi-organization`, key read from a private file outside the repo), no bytecode files.
5. Added `scripts/mis/mis_er200_cam.sh`: `scripts/mis/mis_er_cam.sh` with the same
   hyperparameters, run as one process (no SLURM) on our ER-[200-300] data in `data/cam/` and
   saving to `runs/cam/<PROJECT>/<EXP>`.

6. Added `solvers/cam_prior.py`: `CAMPrior` (`--method CAMPrior`), CAM with an RLSA energy prior
   as the base process. Flip logit = RLSA's flip logit (same gradient, top-d threshold and inner
   temperature schedule as `solvers/rlsa.py`) + the network output as a residual, with the output
   head zero-initialised so that the untrained sampler is RLSA. The prior temperature is also
   scaled by tau_train/tau_train(0) (`--prior_fixed` turns this off), and CAM's entropy term
   becomes tau * KL(Ber(p) || Ber(p_prior)). The module docstring has the formulas and the choice
   of kappa. Each checkpoint gets a `<ckpt>.prior.json` sidecar with the prior scale it was
   trained with, which test mode reads back. New flags: `--prior_*`, `--residual_init`.
7. `main.py`: registers `--method CAMPrior` and its flags (three added lines). All flag names are
   new, and every flag used in `scripts/` still resolves as before under argparse prefix matching.
8. Added `tests/test_cam_prior.py` (CPU, run with `$CAM_PYTHON cam/tests/test_cam_prior.py`).
9. `solvers/meta_solver.py` (unsupervised training loop) and `main.py`: `--save_every_epoch` also
   saves `epoch_XXXX.pt` every epoch (CAMPrior writes its `.prior.json` sidecar next to each), so
   training curves can be re-evaluated offline. Default off; `best.pt` handling is unchanged.
10. `solvers/meta_solver.py`: with `--save_every_epoch`, the untrained weights are also saved as
    `epoch_init.pt` before the first epoch (CAMPrior adds its `.prior.json` with ratio 1).
11. `scripts/env.sh`: `cam_require_gpu` refuses any GPU that runs another user's process
    (previously GPU 1 was refused unconditionally).
12. Added `scripts/mis/test_curve_er700.sh` (test-set curve, CPU) and
    `scripts/mis/large_curve_er700.sh` (ER-[9000-11000] generalisation curve: untrained weights
    and every 10th epoch, 200 steps x 20 samples, several evaluations at a time on one GPU).
    It reads the data from `$CAM_ER700_DIR` and accepts any `<group>_e<E>_s<seed>` run name; the
    untrained weights come from `mis_er700_init2/` if present, else from the run directory.
13. `scripts/env.sh`: the cache directory can be moved with `CAM_CACHE` (for other machines).
14. Added `scripts/mis/seeds_er700.sh` (more seeds of CAM tau0=0.1 and CAMPrior prior tau0=1.0 on
    ER-[700-800], same settings as the seed-0 runs) and `scripts/requirements_env.txt` (pip freeze
    of the env used for all runs). With `SET=ablations` it starts the four CAMPrior ablations.
15. `solvers/cam_prior.py`: ablation switches `--prior_policy {on,off}` (prior in the flip logits
    or not) and `--prior_kl_ref {rlsa,uniform}` (KL to the prior or CAM's entropy term); both
    off reproduce CAM (tests/test_cam_prior.py, test 5). Defaults keep the original behaviour.
16. `solvers/cam_prior.py`: at test time, prior settings that differ from the checkpoint's
    `.prior.json` are an error (previously a warning, so a missing `--prior_d` silently evaluated
    with the default); `--prior_allow_mismatch` keeps the old behaviour on purpose.
17. Added `scripts/mis/mis_rb_cam.sh`: CAM on RB-[200-300] / RB-[800-1200] (`SCALE=small|large`)
    with the flags of the public `mis_rb_small_cam.sh` / `mis_rb_large_cam.sh`, data from
    `$CAM_RB_DIR` (default `/data/shengyuf/MIS`, byte-identical to the HF release), one process per
    GPU named in `CUDA_VISIBLE_DEVICES`, and `MODE=test DEVICE=cpu` for CPU evaluation.
18. Added `scripts/mis/rb_runs.sh` (launches CAM / CAMPrior RB runs with the ER protocol: one GPU
    per run, 1/10 of the official epochs; CAMPrior group names carry the prior's tau0 and d) and
    `scripts/mis/rb_curve.sh` (RB test-set curve on GPU or CPU, retries a failed point at most
    twice and waits between passes instead of retrying at once).
19. Added `solvers/rlsa_distill.py` (`RLSADistill`) and the entry point `distill_rlsa.py`: stage 1
    of the distillation warm start. CAM's network (CAM's own initialisation, flip logits = network
    alone) is fitted to RLSA's per-step flip probabilities, with the forward KL
    KL(Ber(RLSA) || Ber(network)) on RLSA rollouts; the teacher is the CAMPrior prior with
    `--prior_*` (outer factor fixed at 1). Validation is CAM's own inference, and checkpoints are
    plain CAM state_dicts, so stage 2 is `main.py --method CAM ... --checkpoint <best.pt>`.
    `distill_rlsa.py` takes main.py's command line with `--method CAMPrior` and swaps in
    `RLSADistill`; `main.py` itself is unchanged.
20. Added `scripts/mis/mis_er700_distill.sh` (stage-1 launcher: `mis_er700_cam.sh`'s launch block
    with `main.py` replaced by `distill_rlsa.py`) and `scripts/mis/distill_warmstart_er700.sh`
    (both stages per seed on one GPU; stage 2 is the CAM tau0=0.1 command of `seeds_er700.sh`
    plus `--checkpoint`).
21. Added `tests/test_rlsa_distill.py` (CPU, run with `$CAM_PYTHON cam/tests/test_rlsa_distill.py`).
22. Added `scripts/mis/eval_queue_er700.py`: one evaluation queue on one GPU for every ER-[700-800]
    run, on both the test and the large graphs, in place of one `large_curve_er700.sh` per split
    and run set. Same command, protocol and output directories as `large_curve_er700.sh`; the run
    list is a file re-read at every pass, and evaluations already running elsewhere are skipped.

23. Added `solvers/sim_labels.py` (simulated training labels: RLSA rollouts from the two candidate
    next states with common random numbers, and `SimLabelPool`, a fault-tolerant multi-GPU worker
    pool that retries failed tasks on other workers, splits on OOM, restarts or drops dead or hung
    workers and falls back to the trainer's device) and `solvers/lin_labels.py` (first-order labels
    along the trajectory). `solvers/cam_prior.py` gained `--label_mode {cam,sim,lin}`,
    `--label_frac`, `--label_gpus`, `--label_timeout` and `--lin_neighbors`; with the defaults the
    original code path runs unchanged. Derivation: `docs/label_correction.md` (repo root).
24. Added `utils/exact_small.py` (exact expected-energy and soft-value tables of the RLSA base
    process over all 2^n states, n <= ~12), `scripts/mis/sim_label_check.py` and
    `scripts/mis/lin_label_check.py` (small-graph checks of the labels against exact values),
    `scripts/mis/labels_er700.sh` (launcher for the label ablation on ER-[700-800]) and
    `tests/test_sim_labels.py` (12 tests; `$CAM_PYTHON cam/tests/test_sim_labels.py`).
25. Added `scripts/mcut/mcut_ba_large_local.sh`: MaxCut BA-[800-1200] from a local copy of the HF
    `MCut.zip` (`$CAM_MCUT_DIR`), hyperparameters of `scripts/mcut/mcut_ba_large_cam.sh`, the
    runner conventions of `scripts/mis/mis_er700_cam.sh` (one launch per GPU set, `MODE=test`).
26. Added `envs/mcl.py` and `datasets/mcl.py` (`--problem mcl`, maximum clique; `main.py` already
    listed the choice but upstream ships no env for it). As in RLSA (Feng & Yang, 2025, eq. 19) the
    energy is the MIS penalty energy of the complement graph, computed without building it:
    A_c x = (graph sum of x - x)/2 - A x with the dataset's 1/2 edge weights. The model still
    passes messages on the original graph. The continuous decoder is a greedy clique by
    probability; the discrete one is the same local search as `envs/mis.py`. `datasets/mcl.py`
    reuses the MIS loader (same RB graphs). `solvers/cam_prior.py` refuses `--label_mode sim|lin`
    for problems other than MIS (both rebuild the MIS energy from the edge list).
27. Added `tests/test_mcl.py` (`$CAM_PYTHON cam/tests/test_mcl.py`; `MCL_TEST_RB=<dir>` adds real
    RB graphs): energy and gradient against `envs/mis.py` on the explicit complement, decoders
    return maximal cliques, RLSA finds a planted clique.
28. `scripts/mis/mis_rb_cam.sh`, `rb_runs.sh`, `rb_curve.sh` take `PROBLEM=mis|mcl` (default mis,
    unchanged); with `mcl` runs go to `runs/cam/mcl_rb_<scale>/`. Added
    `scripts/mis/rb_rlsa_sweep.sh`: RLSA over a (tau0, d) grid on the first 64 validation graphs
    with CAM's budget, several GPUs, to choose the prior and the same-budget RLSA baseline.
29. `solvers/cam_prior.py`: two more training labels. `--label_mode dense --label_lambda lam`
    replaces the terminal flip gradient by a lambda-weighted average of the flip gradients of the
    later states (lam = 1 is CAM's label); `--label_mode nosign` drops CAM's sign factor
    (-1)^{X_t xor X_1} (exact for a state-independent reference, docs/label_correction.md 6). Both
    need `--num_clips 1`; `tests/test_cam_prior.py` tests 6 and 7. `scripts/mis/labels_er700.sh`
    gains `SET=dense|clips|nosign|camnosign`, `scripts/mis/eval_queue_er700.py` their groups.
30. Added `scripts/mis/d3_label_theory.py` (exact small-graph and CRN large-graph checks of when the
    lean label is right) and `scripts/mis/label_tabular.py` (idealised training on all 2^n states
    with exact label expectations). `d3_label_theory.py large --checkpoint` repeats the CRN check
    under a trained CAMPrior policy (prior + residual); `scripts/mis/label_softopt.py` checks the
    labels against the soft optimum of the CAMPrior objective (docs/label_correction.md section 7).
31. Added `envs/mds.py` and `datasets/mds.py` (`--problem mds`, minimum dominating set, the
    DiffUCO energy A sum x + B sum (1 - x_i) prod_{N(i)} (1 - x_j), A = 1, B = 1.01). The energy is
    multilinear, so the flip difference is exactly (1 - 2x) * grad; the gradient comes from cover
    counts (two neighbour sums). Decoders: greedy cover by probability then descent (continuous),
    descent from the rounded state (discrete); both end at minimal dominating sets.
    `tests/test_mds.py` (5 tests, `MDS_TEST_BA=<dir>` adds real BA graphs). `scripts/mds/` has the
    launcher (BA-small / BA-large from the HF `MDS.zip`, MaxCut BA hyperparameters, since public CAM
    has no MDS script) and the RLSA validation sweep.
32. Added `scripts/tsp/prior_gate.py`: a training-free check of a local-search prior for TSP (the
    analogue of the RLSA prior on MIS, where the prior is the label's operator applied to the current
    state). Each step decodes the state (blended with a constant heatmap, i.e. distance-greedy
    insertion) with CAM's merge + 2-opt and moves the edge state toward that tour, with annealed
    random flips; compared with random-state and distance-greedy decodes at the same final 2-opt.
    Needs the Cython merge built in place (`utils/cython_merge`, `python setup.py build_ext --inplace`).
33. `main.py` / `solvers/meta_solver.py`: `--start_epoch E` continues a run from
    `--checkpoint epoch_<E-1>.pt` (temperature schedule from epoch E, the step-based learning-rate
    schedule advanced by the skipped optimizer steps; optimizer moments start fresh) and
    `--stop_epoch S` stops after epoch S-1 while the schedules still run over `--epochs`.
    `solvers/cam_prior.py`: `--label_mode simpol`, simulated labels under the current sampler
    (prior + network; with `--simpol_kl on` the cost-to-go adds tau * KL, i.e. the soft value of the
    current policy, docs/label_correction.md 7.5), computed in-process in chunks of `--simpol_chunk`
    columns; `--label_states s` labels only a fraction s of the stored states (any label mode).
    `tests/test_cam_prior.py` test 8. `scripts/mis/simpol_er700.sh` launches the three-arm continuation.
34. `models/enprocessde.py`: `EncodeProcessDecode` honours `--use_activation_checkpoint` (the solvers
    already passed it; upstream ignored it there) and then recomputes each message-passing layer in
    the backward pass. Same gradients (bit-identical on CPU; on GPU it differs only by the
    nondeterminism of the sparse matmul, which repeated runs without it show too). Needed for
    DiffUCO's full-trajectory loss with 20 trajectories of 50 steps on ER-[700-800] on 11 GB cards.
    `scripts/mis/slul_er700.sh` launches the controlled SL (DIFUSCO) vs UL (DiffUCO) comparison
    with the small GNN (CAM paper appendix B.2; docs/20261116/handoff.md).
    `solvers/meta_solver.py`: `--save_every_epoch` now also works for the supervised solvers
    (`SLSolver`: `epoch_init.pt` and `epoch_XXXX.pt`), for the same per-epoch test curves.
35. `datasets/mis.py`: with `CAM_GRAPH_CACHE=<dir>` the parsed gpickle graphs (edge_index as int16,
    num_nodes, the KaMIS label if any) are cached as `.pt` files, keyed by the data directory path
    and file name. Parsing a dense ER-[700-800] gpickle with networkx takes 1-3 s, so supervised
    training (every training graph each epoch) was data-bound; a cached graph loads in about 2 ms.
    The cached `Data` is field-by-field identical to the uncached one (checked on train and valid
    graphs). Not set, nothing changes. The cache is not invalidated when label files change: delete
    it then. `scripts/mis/build_graph_cache.py` fills it in parallel; `scripts/mis/slul_er700.sh`
    sets it to `$CAM_CACHE/graphs` by default.
36. `scripts/mis/eval_queue_er700.py` also takes the runs of the SL vs UL comparison
    (`slul_er700.sh`, groups `DIFUSCO`, `DiffUCO_bern`, `DiffUCO_ann`): checkpoints from
    `runs/cam/slul_er700/<exp>/`, test split only, results in `runs/cam/slul_er700_test/<exp>/epKKK/`.
    `--last` now defaults to each run's own epoch count (the E in `_e<E>_s<seed>`).
    Groups `DIFUSCO_lr2e-4` and `DIFUSCO_lr1e-3` (the SL learning-rate check) test like `DIFUSCO`.
    Tuning-grid groups `DIFUSCO_cos_lr<lr>` (test with `--diffusion_schedule cosine`) and
    `DiffUCO_{ann,bern}_lr<lr>_tau<tau0>` (test with their noise type); tau0 grid 0.1, 0.2, 0.4.
    Runs named `pool_<exp>` train on the 163,840-graph pool (`docs/20261116/experiment_plan.md`) and
    test like `<exp>`; `DIFUSCO_lr5e-4` is a group too.
37. `scripts/mis/infer_paths.py` takes optional `--diffuco` and `--difusco` checkpoints of the SL vs UL
    runs. DiffUCO uses the same flip-probability sampler as CAM. DIFUSCO reimplements the step of
    `solvers/difusco.py` (`categorical_posterior`, T=1000 linear, cosine inference) with the shared
    uniforms instead of `torch.bernoulli`; its posterior matches the original to 2e-7.
38. `utils/diffusion_schedulers.py`, `solvers/difusco.py`, `main.py`: `--inference_t_start S` makes the
    DIFUSCO inference steps run from noise level S down to 0 instead of from `num_tp` (default: unset,
    identical schedule). With T=1000 linear training noise and the cosine inference schedule, the first
    ~15 of 50 steps keep less than 5% of the signal; this flag tests whether those steps are wasted.
    The initial state stays uniform, slightly off the forward marginal for small S.
39. `datasets/mis.py`: ported from upstream commit `1d3f4ae` (2026-11-07, "Fix MIS label node
    order"). The `.result` labels are indexed by `sorted(G.nodes())`, but `from_networkx` indexes nodes
    by insertion order; the label is now realigned to the insertion order. Here the realignment is done
    before the label is stored in the graph cache, so cached labels are aligned too. On our ER-[700-800]
    graphs insertion order equals sorted order (checked on all valid/test graphs and 300 sampled training
    graphs of the old and new pools), so ER labels and existing caches are unchanged; RB graphs were
    affected.
40. `solvers/difusco.py`: ported from upstream commit `1d3f4ae`: training samples t from 1..T, as
    in DIFUSCO, instead of 0..T (t=0 is the clean label itself, ~0.1% of training samples). DIFUSCO
    runs started before this change (the learning-rate pilot, the 1000-epoch runs) used 0..T.
41. `solvers/meta_solver.py`: validation also logs the solution size with its natural sign
    (`eval/mis_size` for MIS, `eval/clique_size` for max clique, `eval/cut_size` for max cut, i.e.
    minus `eval/object`), and a test run writes an extra line `Avg. size: S` after `Avg. time`.
    `eval/object` and `Avg. obj` are unchanged: best.pt selection and our log parsers use them.
42. `utils/heated_forward.py`, `solvers/difusco.py`, `main.py`: energy-heated DIFUSCO forward.
    `--heat_lambda L` (default 0 = unchanged) flips node i at step s with probability
    beta_s * sigmoid(lambda_s * dE_i), dE_i the MIS energy change of the flip (same energy as
    `envs/mis.py`); lambda_s decays linearly to 0 at `--heat_until` * num_tp (default 0.5), after which
    the steps are uniform and done in one closed-form jump. x_t is simulated from the label step by step.
    `--difusco_sampler renoise` (default `posterior` = unchanged) replaces the D3PM posterior at
    inference by: draw x0 ~ Bernoulli(predicted probs), run the same forward from x0 to the next step,
    so inference sees the training marginals. Tests: `tests/test_heated_forward.py`; diagnostic:
    `scripts/mis/heated_forward_curve.py`.
43. `utils/bridge_paths.py`, `solvers/fm.py`, `main.py`, `scripts/mis/build_start_bank.py`: FM from an
    arbitrary start distribution. `--fm_bank_train/--fm_bank_eval` give banks of start states (final
    RLSA chain states, built by `build_start_bank.py`); training draws x0 from the bank instead of
    Bernoulli(1/2) and uses the mixture path (x_t = x1 w.p. t, else x0), inference starts each chain from
    its own bank state and takes the same Euler step as before. `--fm_bridge_lambda > 0` replaces the
    mixture path by the bridge of a uniform-flip reference with rate lambda (closed form per node) and
    the step by the Doob-transformed rate. Without these flags FM is unchanged. Tests:
    `tests/test_bridge_paths.py`.
44. `utils/label_sets.py`, `solvers/difusco.py`, `main.py`, `scripts/mis/build_solution_sets.py`:
    several optimal labels per training graph. `build_solution_sets.py` keeps the distinct greedy-decoded
    sets of the best size from many long RLSA chains (label first when it is as large). `--label_set F`
    makes DIFUSCO training pick one label per graph and step with `--label_choice`: `first` (a
    precomputed choice), `random`, or `nearest` (the set most likely under the model's prediction from
    pure noise, a hindsight / hard-EM choice). Validation and test are unchanged. Without `--label_set`
    DIFUSCO is unchanged. Tests: `tests/test_label_sets.py`.
45. `utils/graph_weights.py`, `solvers/difusco.py`, `main.py`: per-graph weights of the DIFUSCO loss,
    sum_i w_g(i) ce_i / sum_i w_g(i). `--graph_weight nsets` uses w = M^-p (M optimal sets of the graph in
    `--graph_weight_sets`, p = `--graph_weight_power`); `--graph_weight ce` follows the hardness-weighted
    curriculum of Zhang et al. (AAAI 2022, eq. 12): w = exp(z), z the standardized log of the graph's
    cross-entropy relative to the running mean at the same noise level, averaged over its visits and
    shared across processes. Logs `train/weight_ess_frac`. Without `--graph_weight` DIFUSCO is unchanged.
    Tests: `tests/test_graph_weights.py`.
46. `scripts/mis/ul_labels.py`: labels from a trained DiffUCO checkpoint. Runs its usual test sampler on
    the training graphs and keeps the largest greedy-decoded set of each graph among its `--num_k`
    samples, written as a one-set-per-graph label file for `--label_set F --label_choice first`.
    `scripts/mis/polish_labels.py` brings such labels up to the KaMIS size where they fall short: RLSA
    chains started at the label, else the nearest set of an optimal-set pool, else the KaMIS label.
47. `utils/label_sets.py`, `solvers/difusco.py`, `main.py`, `scripts/mis/ul_samples.py`: distillation of an
    unsupervised solver's sample distribution. `ul_samples.py` keeps all samples of a DiffUCO checkpoint per
    graph; `--label_choice tilted --label_temp T` draws, each training step, one stored set with probability
    proportional to exp(size / T) (T -> inf: uniform over the samples; T -> 0: the largest).
48. `scripts/mis/sl_samples.py`: all samples of a trained DIFUSCO checkpoint per graph (e.g. with 400 test
    steps), to build labels for distilling many-step sampling into few-step sampling.
49. `scripts/mis/label_learnability.py`: learnability of candidate labels under a reference DIFUSCO checkpoint,
    D(y) = the mean over the trained time steps of the node cross-entropy of the model on q(x_t | y).
50. `scripts/mis/reach_labels.py`: labels from a DIFUSCO student's own samples at the test step budget
    (`sl_samples.py` output): the medoid of each graph's largest samples, and its union with a reference label.
51. `scripts/mis/label_probe.py`: dataset-level learnability probe of a label set: a fresh network trained to
    predict the labels in one step from an uninformative input on part of the training graphs; held-out
    cross-entropy and greedy size on the rest.
52. `scripts/mis/probe_select.py`: per training graph, the M most predictable of the candidate largest sets under a
    two-fold cross-fitted one-step probe trained on reference labels; written for `--label_choice random`.
53. `scripts/mis/diverse_labels.py`: per training graph, all unique largest samples of a solver, or M of them chosen
    by farthest-point sampling in Hamming distance, for `--label_choice random`.
54. `scripts/mis/typical_labels.py`: learnability-thresholded labels: per training graph, the largest size s* whose
    typicality (fraction of a few-step sampler's samples of size >= s*) is at least tau, and the sampler's own
    samples of size >= s* with their multiplicities, for `--label_choice random`.
55. `scripts/mis/strong_labels.py`: labels from a stronger solver filtered by learnability: the largest sets of
    {few-step sampler samples} u {stronger solver's solutions}, the K nearest (Hamming) to the sampler's samples,
    thinned to M by farthest-point sampling; control: farthest-point M of all of them.
57. `scripts/mis/coverage_labels.py`: labels a trained student has not learned yet: of the sampler's largest
    samples, the K farthest (Hamming) from the student's own samples on the training graphs, thinned to M by
    farthest-point sampling; control: the K nearest.
56. `--label_choice soft` (`utils/label_sets.py` `posterior_mean`, `solvers/difusco.py` `soft_targets`): draw a set
    of the graph's label set, noise it to x_t, and train on the posterior mean over the whole set,
    E[x_0 | x_t] = sum_m w_m s_m with w_m proportional to q(x_t | s_m), as a soft cross-entropy target. Same
    minimiser as `random`, lower-variance targets (the per-graph exact Stable Target Field, arXiv 2302.00670).
58. `scripts/mis/kamis_multi.py`: many KaMIS labels per graph (redumis with K seeds; distinct sets of the best
    size, plus the existing label when as large), in the label-set format for `random` / `soft`.
    `scripts/mis/match_label_counts.py`: per graph, every source keeps a random subset of as many sets as the
    smallest source has, so source comparisons are not confounded by the number of labels.
59. `--difusco_sampler renoise_decode | renoise_feasible` (`solvers/difusco.py` `decode_x0`): in the re-noising
    sampler, make the sampled x0 feasible with the problem's own greedy decoder before noising it to t-1
    (sampled nodes first): `renoise_decode` keeps the maximal decoded solution, `renoise_feasible` only drops
    conflicting sampled nodes. Diagnostic `CAM_TRACE_INPUTS=FILE` (`trace_inputs`): per step, the sampler's
    input vs a training-form input (the chain's final solution noised to the same t).
60. Edge problems (TSP) with DIFUSCO: `models/angnn.py` activation-checkpoint branch read a non-existent
    attribute (`node_feature_only`); it now uses the same `add_time_on_edge` rule as the plain branch (and
    `use_reentrant=False`), so `--use_activation_checkpoint` works for the 12x256 TSP-500 network on 11 GB GPUs.
    `solvers/difusco.py` `renoise` handles edge variables: x0 ~ Bernoulli(p) per edge, then the plain training
    forward `CategoricalDiffusion.sample` to t-1 (the decode variants stay node-only).
61. Several tours per TSP instance as DIFUSCO labels (`utils/tour_sets.py`; `solvers/difusco.py` `edge_slices`,
    `tour_labels`, edge branches of `pick_labels` / `soft_targets`): with `--label_set` a tour-set file (per
    instance the distinct tours, sorted by length; checked against the points of `--train_path`), every step uses
    one tour (`random`, `first`) or the posterior mean over the set (`soft`). Tours become undirected edge labels
    on the sparse kNN graph (edge (i, j) is 1 when j follows or precedes i), unlike the upstream directed
    i -> successor(i) label. Tools in `scripts/tsp/`: `gen_instances.py` (uniform instances), `lkh_labels.py`
    (LKH-3 with K seeds), `dump_tours.py` (all tours a trained model samples, any `--method`),
    `match_tour_sets.py` (equal numbers of tours per instance across sources). Note: the TSP dataset reads
    `lines[:num_samples]`, so the default `-1` drops the last instance; pass `--test_samples` explicitly.
62. `--difusco_sampler renoise_decode` on TSP (`solvers/difusco.py` `decode_tour`, `--decode_two_opt N`): at every
    step the sampled edges x0 are merged into a tour by the env's greedy edge insertion (sampled edges first), then
    N iterations of 2-opt (0: none), and the tour (undirected edge labels) is noised to t-1 by the training forward.
    `scripts/mis/ul_samples.py` also dumps the samples of a CAM checkpoint (same flip sampler as DiffUCO).
63. `--ul_chain_decode` (`solvers/meta_solver.py` `ULSolver.chain_decode_tour`): the same in-chain decoding for the
    UL flip sampler (CAM, DiffUCO) on TSP: after every non-final step the current edges are merged into a tour
    (sampled edges first, then by their post-flip probability), `--decode_two_opt N` iterations of 2-opt, and the
    tour's undirected edge labels are the state for the next step.
64. `datasets/mcut.py` (also the MDS dataset): a node label is read from `anno/<stem>.npy` only where that file
    exists, otherwise `x` is zeros. MDS on the RB graphs reuses `/usr1/shengyuf/MIS/RB_small_*`, whose `anno/`
    holds MIS labels; supervised MDS (DIFUSCO) takes its labels from `--label_set` (`scripts/mds/ilp_labels.py`).

`scripts/config.sh` and the other upstream scripts are unchanged and not used by our runner.
`CAMPrior` runs through the same runner by appending flags, e.g.
`CUDA_VISIBLE_DEVICES=<gpu> EXP=camprior bash cam/scripts/mis/mis_er200_cam.sh --method CAMPrior`.

## Data

`scripts/export_cam_dataset.py` (repo root) converts `data/datasets/<name>/{train,val,test}.npz`
into CAM's gpickle layout under `data/cam/<name>/ER_{train,valid,test}/` (gitignored).
