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
| 1D 学出来的配对 | 未开始 | | | | |
| 1E 标签变体 | 未开始 | | | | |
| 1F 加固 | 未开始 | | | | |
| 2A 迁移单图 DASBS | 通过 | 11-25 09:55 | 11-25 10:25 | 迁移后 16 项测试通过；15 个阶段后总变差 path6 0.092 ± 0.021、cycle8 0.135 ± 0.023（标准 0.09 / 0.14 ± 0.05），path6 三个种子与 11-15 完全相同；修正项改在 t = 0 回归：0.085 / 0.161，差别在种子间波动之内 | `experiments/20261125_dasbs_smallgraph/README.md` |
| 2B 跨图共享，小图验证 | 未开始 | | | | |
| 2C RB-small 训练 | 未开始 | | | | |
| R 总报告 | 未开始 | | | | |

## 正在跑的任务

| 任务 | 进程号 | GPU | 日志 | 预计结束（UTC） |
|---|---|---|---|---|
| 1B 基线第二种子：独立配对随机抽（indep_s1） | 1674149 | 4,5 | `runs/experiments/20261125_sbdifusco_rbsmall/20261125_sbdifusco_rbsmall_indep_s1/train.log` | 11-25 11:30 |
| 1B 基线第二种子：独立配对软目标（indep_soft_s1） | 1674186 | 6,7 | 同目录 `..._indep_soft_s1/train.log` | 11-25 11:30 |
| 1B 回归训练：SB 代码路径 ω = 0（sb_w0_soft_s0） | 1689431 | 0 | 同目录 `..._sb_w0_soft_s0/train.log` | 11-25 13:45 |
| 1C ω = 0.35 种子 0（sb_w0.35_soft_s0） | 1689459 | 3 | 同目录 `..._sb_w0.35_soft_s0/train.log` | 11-25 13:45 |
| 1C ω = 0.5 种子 0（sb_w0.5_soft_s0） | 1694537 | 1 | 同目录 `..._sb_w0.5_soft_s0/train.log` | 11-25 14:00 |
| 1C ω = 1.0 种子 0（sb_w1.0_soft_s0） | 1705400 | 2 | 同目录 `..._sb_w1.0_soft_s0/train.log` | 11-25 14:10 |
| 验证集前 64 张图的 KaMIS 多解（1C 的起点依赖分析用） | 1708059 | 无（CPU 16 进程） | `/usr0/home/shengyuf/yunyi/data/rb_small_sb/val64_kamis.log` | 11-25 11:00 |

## 待用户确认

（无）

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
