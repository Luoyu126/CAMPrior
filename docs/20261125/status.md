# 进度：薛定谔桥用于 RB-small MIS（按 `plan.md` 执行）

上下文被压缩后先读本文件，再读 `plan.md`。

状态只有六种：未开始、进行中、通过、失败、阴性、跳过（后四种是终态）。

## 阶段表

| 阶段 | 状态 | 开始（UTC） | 结束（UTC） | 结论 | 证据位置 |
|---|---|---|---|---|---|
| M1 路线 1 推导与验证 | 通过 | 11-25 | 11-25 09:15 | 推导全部成立；20 项测试通过（误差 < 1e-8，p = 0.5 与现有软目标 < 1e-6，反向对照误差 0.008–0.16）；两个子 agent 复核，对抗复核的 1 个错误（采样器 B 配乘积型网络时偏差不随步数消失）已改正 | `docs/20261125/math_route1.md`（§9 验证表、§10 复核）、`cam/tests/test_sb_coupling.py`、`runs/scratch/m1_independent/`、`runs/scratch/m1_review/` |
| M2 路线 2 推导与验证 | 通过 | 11-25 | 11-25 10:08 | 推导全部成立；`test_dasbs_exact.py` 33 项 + `test_sb_cross.py` 2 项 + 2A 的 16 项全部通过（逐项相对误差 1e-10）；反向对照扩到 3 张图 × 2 种噪声表均偏离；独立推导逐项吻合；对抗复核的 3 个文字错误（§12 符号、§12 描述、§9 数字）已改，17 个缺口已补测试或写明 | `docs/20261125/math_route2.md`（§13 验证表、§15 复核）、`cam/tests/test_dasbs_exact.py`、`docs/20261125/m2_experiments.txt`、`runs/scratch/m2_independent/`、`runs/scratch/m2_review/` |
| M3 交叉验证 | 通过 | 11-25 09:40 | 11-25 09:42 | 路线 2 精确解（ν = 标签上的 π1）与路线 1 的 Sinkhorn 解一致：配对、6 个时刻的边缘、标签后验误差均 ≤ 1.1e-12（< 1e-8），12 个实例 × 2 种噪声表 | `cam/tests/test_sb_cross.py`；`math_route2.md` §11 |
| 1A 配对分析 | 通过 | 11-25 09:14 | 11-25 09:45 | K = 2^22 时标签边缘最大相对误差 ≤ 1.10%（< 2%）；4000 张图预计算最多 18.3 分钟；给 1C 的 ω = 0.35、0.5、1.0（有效标签数中位数 18.3、11.7、4.1）；阴性出口未触发 | `experiments/20261125_sbdifusco_rbsmall_coupling/README.md` |
| 1B 实现与回归测试 | 进行中 | 11-25 09:20 | | 标准 1（单元测试 + 原有 10 个测试文件全部通过）、2（ω = 0 时 200 步损失与现有软目标逐步相同，差 0）、3（两种采样器代码路径终点等于 π1，卡方检验通过）已满足；标准 4 的回归训练进行中 | `experiments/20261125_sbdifusco_rbsmall/README.md`、`cam/scripts/sb/check_difusco_sb_sampler.py` |
| 1C 只用多解，扫 ω | 进行中 | 11-25 09:35 | | 第一个种子的 3 个 ω 训练中 | `experiments/20261125_sbdifusco_rbsmall/` |
| 1D 学出来的配对 | 进行中 | 11-25 10:35 | | 前置完成：CSBM 已下载，要点写入 math_route1.md §12；验收标准 1（表格 D-IMF 收敛到 Sinkhorn 配对，TV < 1e-3）已满足，6 个实例全部通过；训练部分要等 1C 选出 ω | `docs/20261125/math_route1.md` §12、`cam/utils/sb_imf.py`、`cam/tests/test_sb_imf.py` |
| 1E 标签变体 | 进行中 | 11-25 10:28 | | 前置分析完成：选定扰动为"翻转 10% 的位后贪心修复成极大独立集"；加权 π1 的代码与测试已提交；训练要等 1C 选出 ω | `runs/experiments/20261125_sbdifusco_rbsmall_labelvar/stats_*.txt`、`cam/scripts/sb/perturb_labels.py`、`cam/tests/test_sb_weights.py` |
| 1F 加固 | 未开始 | | | | |
| 2A 迁移单图 DASBS | 通过 | 11-25 09:55 | 11-25 10:25 | 迁移后 16 项测试通过；15 个阶段后总变差 path6 0.092 ± 0.021、cycle8 0.135 ± 0.023（标准 0.09 / 0.14 ± 0.05），path6 三个种子与 11-15 完全相同；修正项改在 t = 0 回归：0.085 / 0.161，差别在种子间波动之内 | `experiments/20261125_dasbs_smallgraph/README.md` |
| 2B 跨图共享，小图验证 | 进行中 | 11-25 10:15 | | 标准 1（memoryless 特例与 M2 一致）已在单元测试中满足；三组训练（sb、ml、nocorr）在 CPU 上进行中 | `experiments/20261125_dasbs_crossgraph/`、`cam/tests/test_dasbs_solver.py` |
| 2C RB-small 训练 | 未开始 | | | | |
| R 总报告 | 未开始 | | | | |

## 正在跑的任务

| 任务 | 进程号 | GPU | 日志 | 预计结束（UTC） |
|---|---|---|---|---|
| 1C ω = 0.35 种子 1（sb_w0.35_soft_s1） | 2820987 | 5（与 1E 对照共用） | `runs/experiments/20261125_sbdifusco_rbsmall/20261125_sbdifusco_rbsmall_sb_w0.35_soft_s1/train.log` | 11-25 19:00 |
| 1C ω = 0.5 种子 1（sb_w0.5_soft_s1） | 2821018 | 6（共用） | 同目录 `..._sb_w0.5_soft_s1/train.log` | 11-25 19:00 |
| 1C ω = 1.0 种子 1（sb_w1.0_soft_s1） | 2821054 | 7（共用） | 同目录 `..._sb_w1.0_soft_s1/train.log` | 11-25 19:00 |
| 基线评测队列（两个基线 × 两个种子：锚点 50/20/10/5 步 × 两种采样器，另有成对追踪） | 2453475（队列） | 4 | `runs/experiments/20261125_sbdifusco_rbsmall/queues/after_baselines_s1.log` | 约 14:00 |
| 1B 回归训练：SB 代码路径 ω = 0（sb_w0_soft_s0） | 1689431 | 0 | 同目录 `..._sb_w0_soft_s0/train.log` | 11-25 15:15（与 1E 共用 GPU 后每 epoch 47→59 秒） |
| 1C ω = 0.35 种子 0（sb_w0.35_soft_s0） | 1689459 | 3 | 同目录 `..._sb_w0.35_soft_s0/train.log` | 11-25 15:20 |
| 1C ω = 0.5 种子 0（sb_w0.5_soft_s0） | 1694537 | 1 | 同目录 `..._sb_w0.5_soft_s0/train.log` | 11-25 15:30 |
| 1C ω = 1.0 种子 0（sb_w1.0_soft_s0） | 1705400 | 2 | 同目录 `..._sb_w1.0_soft_s0/train.log` | 11-25 15:45 |
| 自动接续：SB 第一种子各自训练结束后，在原 GPU 上评测（锚点 50/20/10/5 步 × 两种采样器；成对追踪） | 2559895（4 个子 shell 各等一个 GPU）| 0–3（接续后） | `runs/experiments/20261125_sbdifusco_rbsmall/queues/after_sb_s0.log` | 训练 15:15–15:45 结束，评测各约 20 分钟 |
| 2B 自动接续：各 run 写出 epoch_0047.pt 后跑精确评测（test、valid、wl 三组并行，各 4 线程，CPU） | 2583343 | — | `runs/experiments/20261125_dasbs_crossgraph/after_train.log`；结果 `exact_<set>_epoch_0047.jsonl` | 12:50 开始，约 1 小时，~13:50 |
| 1E 势的预先计算（ω 未定，先算 0.35/0.5/1.0 三个候选；单解 + 扰动 等权/T=1/T=0.5，k128 等权） | 2584159、2584160、2584161 | 0、1、2（与训练共用） | `runs/experiments/20261125_sbdifusco_rbsmall_labelvar/precompute_spec_w*.log` | 单解部分已完成；k128 约 12:15 |
| 1E 对照：单解 + 扰动，等权，独立配对（ω = 0） | 2588218 | 3（与 1C 共用） | `runs/experiments/20261125_sbdifusco_rbsmall_labelvar/20261125_sbdifusco_rbsmall_labelvar_single_w0_s0/train.log` | 11-25 18:00 |
| 1E 对照：单解 + 扰动，加权 T = 1，独立配对 | 2602325 | 0（共用） | 同目录 `..._single_w0_T1_s0/train.log` | 11-25 18:30 |
| 1E 对照：单解 + 扰动，加权 T = 0.5，独立配对 | 2602356 | 1（共用） | 同目录 `..._single_w0_T0.5_s0/train.log` | 11-25 18:30 |
| 1E 对照：k128 多解，等权，独立配对 | 2602390 | 2（共用） | 同目录 `..._k128_w0_s0/train.log` | 11-25 18:30 |
| 1E 势的预先计算（续）：多解 + 扰动 等权/T=1/T=0.5，ω = 0.35/0.5/1.0 | 2740681、2740682、2740683 | 0、1、2（共用） | `runs/experiments/20261125_sbdifusco_rbsmall_labelvar/precompute_spec_mkamis_w*.log` | 约 14:30 |
| 1E 对照：多解 + 扰动，等权，独立配对 | 2740382 | 5（共用） | `runs/experiments/20261125_sbdifusco_rbsmall_labelvar/..._mkamis_w0_s0/train.log` | 11-25 19:00 |
| 1E 对照：多解 + 扰动，加权 T = 1，独立配对 | 2740410 | 6（共用） | 同目录 `..._mkamis_w0_T1_s0/train.log` | 11-25 19:00 |
| 1E 对照：多解 + 扰动，加权 T = 0.5，独立配对 | 2740438 | 7（共用） | 同目录 `..._mkamis_w0_T0.5_s0/train.log` | 11-25 19:00 |
| 1D 流程冒烟测试（E0 = E = 1，L = 1，J = 2，种子 99，不进 wandb，结果不用） | 2618277 | 3（共用） | `runs/experiments/20261125_sbdifusco_rbsmall_imf_smoke.log` | 约 13:00 |
| 2B sb：log-linear α = 0.5，带修正项，种子 0 | 2415828 | 无（CPU 8 线程） | `runs/experiments/20261125_dasbs_crossgraph/20261125_dasbs_crossgraph_sb_s0/train.log` | 11-25 12:50 |
| 2B ml：memoryless（α = 0），无修正项，种子 0 | 2415840 | 无（CPU 8 线程） | 同目录 `..._ml_s0/train.log` | 11-25 12:50 |
| 2B nocorr：α = 0.5 但不用修正项（对照），种子 0 | 2415852 | 无（CPU 8 线程） | 同目录 `..._nocorr_s0/train.log` | 11-25 12:50 |

## 待用户确认

- 1D 验收标准 2（"学到的配对代价与闭式配对相差 < 10%"）偏弱：ω = 0.5 时独立配对本身只差 3.1%。按原标准记录，另报"缩小差距的比例"。见执行记录 11-25 10:50–11:05。

## 执行记录

- 11-25：开始执行。GPU 0–9 全部空闲。
- 11-25 09:15：M1 通过。对 1A/1B 的影响（记在这里，执行时照办）：
  - 1A：蒙特卡洛 Sinkhorn 在 M 大、ω ≥ 1 时需要 K 到 10^6 量级才能满足相对误差 < 2%；要按 ω 报告所需 K，测量用的新起点数要远大于 K（math_route1.md §7）。
  - 1B：接入 DIFUSCO 需要 heat_lambda = 0、每张图一个 log_g、采样器 B 要把起点 z 传进 renoise；采样器 A 直接用现有 categorical_posterior + `signal_to_Q_bar(a, as_numpy=True)`。
  - 1C：采样器 B（含现有 renoise）配乘积型网络有不随步数消失的偏差，A 没有；两种都要评测（math_route1.md §5）。
- 11-25 09:50：1A 主表完成；启动 ω = 1 补跑和两条基线的第二种子。验证锚点：`/usr0/home/shengyuf/yunyi/data/rb_small_valanchor`（验证图加载顺序第 64–499 张的符号链接）。
- 11-25 09:14–09:45：1A 完成并提交（主表 + ω = 1 用 K = 2^22 补跑；三个 ω 的 g 全部预计算完，`/usr0/home/shengyuf/yunyi/data/rb_small_sb/`）。
- 11-25 09:20–09:35：1B 实现完成并提交（`--sb_coupling --sb_omega --sb_logg`，UPSTREAM.md 第 69 条）。
  - 标准 2 的做法：两次独立运行无法逐步对比，因为现有实现本身在 GPU 上不确定（两次相同的现有实现，第 1 步差 2.4e-7，第 4 步差 0.05，来自 scatter_add 的原子加）。
  - 改为在同一次运行里，对同一批 x_t 和同一 logits 同时算两种软目标的损失。200 步全部相等（差 0）。
  - 注意：两次早期的检查运行忘了关 wandb，被同步到 SLVUL 项目（run 名 dulcet-leaf-2 等，均为 1 epoch、2000 张图的短跑）。之后检查运行都设 WANDB_MODE=disabled。
- 11-25 09:35–09:46：启动 1B 回归训练（ω = 0）和 1C 第一个种子（ω = 0.35、0.5、1.0）。单卡、batch 40；基线用两卡各 20，有效 batch 相同。
- 11-25 09:42：M3 通过。
- 11-25 09:48：math_route2.md 草稿完成；M2 的两个子 agent 复核进行中。
- 11-25 09:52：提交起点依赖诊断（`CAM_SB_PAIR_TRACE`：相邻两条链共享起点 z，记录起点与最终解；分析脚本 `cam/scripts/sb/start_dependence.py`）。CPU 小测试通过。
- 11-25 09:55–10:05：2A 代码迁移。在原工作树的 Python 环境下用同一小配置（path6，2 阶段）分别运行原代码和迁移后的代码，训练日志和评测指标逐位相同。修正（按 M2 结论）以选项形式加入、默认关闭：修正项回归改在 t = 0（math_route2.md §5：t 接近 1 时比值方差大）；采样时间网格在 t = 1 附近加密（§10）。复现时两种修正项时间都跑。
- 11-25 10:08：M2 通过。两个子 agent 的结论写入 math_route2.md §15。对后续阶段的影响（执行时照办）：
  - 2B：GNN 的区分能力受 1-WL 限制，小图族里放 WL 不可区分的图对作诊断；修正项输出可限制在 [tanh(Γ01/2), coth(Γ01/2)]；测量 corr-DM 比值场的回路不一致。
  - 2C：标签在对数空间算、float64、不截断；报告原始终点可行比例与解码前后结果；阶段数和起点分布要当作待验证的量（DASBS 低温下不用均匀 μ）；模型参数 EMA 与样本缓冲区参照 DASBS 附录 C。
- 11-25 10:10–10:25：2A 复现完成，通过。
- 11-25 10:15–10:35：2B 代码完成并提交：`cam/solvers/dasbs.py`（`--method DASBS`，UPSTREAM.md 第 71 条）、小图族生成 `cam/scripts/sb/gen_small_graphs.py`（数据在 `/usr0/home/shengyuf/yunyi/data/sb_small/`：训练 2000、验证 64、留出测试 64，n = 8–12；另有 6 对 1-WL 不可区分的图）、精确评测 `cam/scripts/sb/dasbs_exact_eval.py`（逐步传播采样器的精确分布，不抽样）。测试 `cam/tests/test_dasbs_solver.py` 5 项通过，含 2B 标准 1（memoryless + 修正项恒为 1 时与 M2 的特例一致）。
- 11-25 10:25：验证集前 64 张图的 KaMIS 多解完成（`/usr0/home/shengyuf/yunyi/data/rb_small_sb/val64_kamis_k128.pt`，平均每图 69.8 个解，大小 20.547，按加载顺序）。
- 11-25 10:25：启动 2B 三组训练（48 个 epoch × 125 次迭代 = 20 个阶段）。GPU 已有 8 张在用（上限），所以放在 CPU 上。提交 1C 的配对比较脚本 `paired.py` 与评测队列 `eval_queue.sh`。
- 11-25 10:28–10:35：1E 前置分析（不训练，400 张训练图 × 每个标签 32 次扰动，`cam/scripts/sb/perturb_labels.py stats`）：
  - 等大小交换（1 步）：大小不变，与原解 Hamming 2，每图只有约 13 个不同结果，95% 不在 KaMIS 多解里（是新的最优解）。只换 1 步几乎等于"多给几个最优解"，没有质量差别，加权无意义。
  - 删 1–2 个点：平均小 1.5，Hamming 1.5。只是"缺点的最优解"，不是自然的局部解。
  - 翻转 10% 后贪心修复：43% 与原解同样大，32% 小 1，17% 小 2，8% 小 3 及以上；Hamming 平均 6.1，每 32 次有 26.8 个不同结果。既有多样性，又有质量差别，加权（T = 1、0.5）才有意义。**选这一种。**（翻转 5%：83% 同样大、Hamming 2.1，差别太小。）
  - 标签集：单解 + 32 次扰动（每图约 28 行）；多解 + 每个解 2 次扰动（每图约 120 行）。文件名带 `_loaderorder`，节点顺序与加载器一致。
- 11-25 10:30：加权 π1 的支持：π1(m) ∝ exp(|s_m| / T)。ω = 0 时 log g = log π1（`precompute_logg.py --omega 0 --weight_temp T`，训练用 `--sb_omega 0 --sb_logg`）；ω > 0 时是同一个 Sinkhorn、换目标边缘。测试 `cam/tests/test_sb_weights.py` 3 项通过（精确 Sinkhorn 边缘误差 < 1e-12，蒙特卡洛版本 < 1e-5 相对误差，ω = 0 时软目标等于以 π1 为先验的贝叶斯后验，误差 < 1e-12；两个反向对照偏离 > 1e-3）。
- 11-25 10:35：下载 CSBM（`ref/lit_pdfs/2502.01416.pdf` 与 `.txt`），1D 前置的文献要点由子 agent 整理中。
- 11-25 10:35：1C 评测准备：`eval_queue.sh`（按列表依次评测，已有结果的跳过）、`evaluate.sh` 增加 `val64`（验证集前 64 张，配 KaMIS 多解做起点依赖分析）和 `PAIR_TRACE=1`（成对起点追踪，也用来算单样本平均）。基线第一种子的 checkpoint 以符号链接放在 `runs/experiments/20261125_sbdifusco_rbsmall/baseline_indep{,_soft}_s0/`，不往 11-19 的目录里写文件。
- 11-25 10:40–10:48：1D 前置。
  - 精确的表格 D-IMF（`cam/utils/sb_imf.py`）在 6 个小实例上都收敛到 Sinkhorn 配对（TV < 1e-3；ω ≤ 1 时 4–19 次投影，ω = 2、3 时 43、70 次）。到 SB 的 KL 单调不增。
  - 两个反向对照都偏离：用前向边缘代替桥时 TV > 0.02；换一个 ω 时差距 > 1e-2。
  - 测试 `cam/tests/test_sb_imf.py` 12 项通过。
  - 文献要点写入 math_route1.md §12。两篇论文都没有"两个方向同时训练"，1D 采用"交叉喂入"（§12.4，属于我们的推断）。
- 11-25 10:40–10:47：2C 代码准备（2B 未通过前不开训练）。
  - **问题**：RB-small 图的度数最高约 70，初期随机 rollout 的 ΔH 可到 −80，τ = 0.1 时比值 exp(−ΔH/τ) 在 float64 中溢出。CPU 检查：从头训练时 16 个控制器批次全部溢出。
  - **因此 2C 的温度不能照搬 DiffUCO 的 0.1 → 0**，改为从 1 线性退到 0.05。这是方法本身的限制（目标是 exp(−H/τ)，τ = 0 没有定义），不是放宽标准。
  - **新增选项**（默认关闭，2B 不受影响）：
    - 控制器输出除以 τ；
    - 损失按 max(1, 预测比值) 归一，驻点不变，有测试；
    - 溢出批次跳过并计数，日志里报告比例；
    - 参数 EMA 0.999。
  - **评测**额外记录采样器原始终点的可行比例和大小（解码前后）。
  - 测试 `cam/tests/test_dasbs_solver.py` 7 项通过。
  - 启动和评测脚本已提交：`experiments/20261125_dasbs_rbsmall/`。
- 11-25 10:50–11:05：1D 代码。
  - 后向网络 `--sb_reverse`：信号表反转为 b_k = a_T / a_{T−k}，预测起点 z。
  - 配对缓存训练 `--sb_pair_cache`：软目标是缓存配对构成的经验配对下的精确后验。
  - 缓存构造 `cam/scripts/sb/imf_cache.py`；迭代脚本 `experiments/20261125_sbdifusco_rbsmall_imf/run_imf.sh`。
  - 测试 `cam/tests/test_difusco_sb_reverse.py` 3 项通过：后向采样器配精确后验时，终点配对服从 SB 配对（卡方 p = 0.93 / 0.69）；用独立配对的后验时偏离（反向对照）。
  - 原有路线 1 的测试（36 项）和采样器检查全部仍然通过。
  - **注意（1D 验收标准 2 偏弱）**：ω = 0.5 时，独立配对的平均 Hamming 代价（113.7）只比闭式 SB 配对（110.3）高 3.1%，所以"与闭式配对相差 < 10%"在第 0 轮就已满足。结果按原标准记录，另外报告"缩小了多少差距"=（独立 − 学到）/（独立 − 闭式）。**待用户确认**这个标准是否要改。
- 11-25 11:05：1E 准备。
  - 单解 + 扰动标签集完成：每图平均 27.5 个，平均大小 19.05；按大小加权后，T = 1 时 19.69，T = 0.5 时 19.97。
  - ω = 0 的加权势已算好（`logg_single_w0_T{1,0.5}.pt`）。
  - 启动脚本与 README（含扰动分析表）已提交：`experiments/20261125_sbdifusco_rbsmall_labelvar/`。
- 11-25 11:10：2B 评测脚本 `experiments/20261125_dasbs_crossgraph/evaluate.sh`（预先指定最后一个 epoch 的 checkpoint，不做选择），训练结束后自动跑。1E 的 ω > 0 势在 1C 出结果前先把三个候选都算上，省去关键路径上的等待；选定 ω 后其余两个不用。
- 11-25 11:15：1E 中与 ω 无关的独立配对对照先开跑（单解 + 扰动 等权 / T = 1 / T = 0.5，k128 等权），每张 GPU 和一个 1C 训练共用。实测：训练 GPU 利用率只有约 30%（瓶颈在 CPU 端），两个训练共用一张卡时原训练每 epoch 由 47 秒变为 59 秒，新训练 59 秒，总吞吐约 1.6 倍。代价是 1C 第一种子推迟约 1 小时出结果（ω 的选择随之推迟）。1E 的 SB 组等 ω 选定后再开。
- 11-25 11:20：1D 冒烟测试：第 0 轮（两个网络在独立配对上各训 1 个 epoch）正常；采样阶段每张训练图约 1 秒，一轮采样约 1 小时（两个网络并行），完整 1D（E0 = 100、E = 50、L = 3）约 8 小时。
- 11-25 11:52：多解 + 扰动标签集完成：每图 116.7 个，期望大小 等权 19.47 / T = 1 19.96 / T = 0.5 20.10（统计表见 1E README）。ω = 0 加权势已算，三个独立配对对照已开跑（GPU 5–7，与基线第二种子共用）。
- 11-25 12:15：基线第二种子（独立配对随机抽 / 软目标）训练完成；1C 三个 ω 的第二种子在 GPU 5–7 启动，GPU 4 开始评测基线。
