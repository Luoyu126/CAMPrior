# Handoff：DIFUSCO 加热前向和采样器对照（2026-11-17 11:00 匹兹堡时间写）

接着 `handoff_1116.md`。用户偏好和仓库规则见 `docs/handoff.md` §0、`AGENTS.md` 和记忆文件。要点：
- 中文回复，内部代号用白话解释；时间一律报匹兹堡时间（America/New_York），日志里是 UTC，要换算。
- 只在 `/usr0/home/shengyuf/yunyi/` 下写文件；`transfer/` 不是我们的，不要动。
- 提交用 Yunyi Chen 的身份，一个模块一个提交，推到 `develop`，不加 AI 署名。
- 只按 pid 杀自己的进程。
- **启动训练前必须先汇报、等用户明确说开跑**；发现异常先汇报。
- **有别人在用卡时不跑**；启动前看 nvidia-smi 和进程属主。
- DIFUSCO 单卡 batch 40 可以（对齐 DiffUCO 的 10 卡 × 4）；本次加热前向两组按用户要求用 4 卡。

## 1. 研究线索（11-16 到 11-17 和用户商定）

- 问题：MIS 上为什么 UL（DiffUCO）好于 SL（DIFUSCO）。
- 用户的假设：SL 只有一个精确标签，相当于只知道能量的一个最低点；UL 在自己的样本上算能量，看到了地形。
- 第一步验证：DIFUSCO 的前向加噪改成按能量往上坡翻（“加热前向”），检验“训练数据带上地形”有没有用。
- 详细计划和讨论：`docs/20261116/experiment_plan.md` §7。页面版（用户逐条改过）：https://claude.ai/artifact/P3mnUGg91NA9T6d32793b4 。旧的“SL vs UL 实验计划”页面已按用户要求删除。

## 2. 加热前向和重新加噪采样器（代码已提交）

- 能量与 `cam/envs/mis.py` 相同：E = Σ x_i(β(Ax)_i + b_i)，β = 1.02，b = −1。翻转差 ΔE_i = (1 − 2x_i)(2β k_i − 1)，k_i 是取 1 的邻居数。
- 前向：节点 i 在第 s 步的翻转概率是 β_s · σ(λ_s ΔE_i)，各节点独立。λ_s 从 λ 线性退到 0，在 t = 500 处退完，之后是均匀翻转（用闭式一步跳过），终点仍是 Ber(0.5)。不控制每步的期望翻转数（用户改定）。λ = 0 时就是均匀核。
- 训练时 x_t 从标签逐步模拟（没有闭式边缘分布）。
- 采样器 `--difusco_sampler renoise`（重新加噪）：网络给出 x̂₀ 概率，按 Bernoulli 抽一个 x̂₀，再用同一条前向链加噪到下一步，不用 x_t。默认 `posterior` 是原来的 D3PM 后验。
- 参数：`--heat_lambda`（默认 0）、`--heat_until`（默认 0.5）、`--difusco_sampler`。
- 文件：`cam/utils/heated_forward.py`，`cam/solvers/difusco.py`，`cam/main.py`，`cam/UPSTREAM.md` 第 42 条，测试 `cam/tests/test_heated_forward.py`（5 项全过），诊断 `cam/scripts/mis/heated_forward_curve.py`。
- 用户定的两条局限：(1) 训练时无法从 x₀ 一步加噪得到 x_t，只能逐步模拟；(2) 推理时得不到后验需要的两个分布，目前用有偏的做法，从 x̂₀ 直接加噪还原 x_{t−1}，而不是从以 x̂₀、x_t 为条件的后验里采样。阶段 2 要解决第 (2) 条（候选：冻结局部场、Cold Diffusion 式共用随机数的差分更新、重要性重采样），还没开始。

## 3. λ 的选择（诊断）

- λ ≥ 0.3 时加热链冲过头：t = 300–500 时密度 0.6–0.9，冲突边是均匀时的 2–3 倍，标签的点几乎全被锁成 1（正反馈）。λ = 1、3、10 已饱和。
- λ = 0.1 峰值密度只到 0.53，用户定只跑 λ = 0.1（加上 λ = 0 对照）。
- 表格见 `experiments/20261116_difusco_rbsmall_heatflip/README.md`。

## 4. 已有结果：训练 × 采样器的 2 × 2 表（11-17，1000 张 test，推理种子 0 / 1）

| best.pt 来自 | D3PM 后验采样 | 重新加噪采样 |
|---|---|---|
| `20261116_difusco_rbsmall_randflip`（第 119 个 epoch） | 19.314 / 19.290 | 19.608 / 19.628 |
| `20261116_difusco_rbsmall_heatflip` λ = 0（第 49 个 epoch） | 19.236 / 19.230 | 19.578 / 19.582 |

- 采样器的影响约 +0.33，两个权重上一致；推理种子间只差约 0.02。
- 参照：DiffUCO bernoulli 19.70 / 19.74，annealing 19.87。换成重新加噪后，和 DiffUCO bernoulli 的差距从约 0.4 缩到约 0.1。
- λ = 0 那组是 patience 20 早停的（第 49 个 epoch 最好，第 69 个 epoch 停），当时学习率还有约 86%。用户认为停得太早、没训稳，所以有了第 5 节的不早停重跑。
- 测试脚本：`experiments/20261116_difusco_rbsmall_heatflip/test.sh <gpu> <randflip|lam0|lam0p1> <posterior|renoise> [test_seed]`，结果写在对应 run 目录的 `test_<sampler>_seed<N>_result.txt`。

## 5. 正在跑的实验（11-17 10:51 匹兹堡时间启动，种子 0，不早停，跑满 200 个 epoch）

| run | GPU | 采样器 | wandb |
|---|---|---|---|
| `20261117_difusco_rbsmall_heatflip_lam0p1_s0` | 1–4，每卡 10 张 | 重新加噪 | 4ossajux |
| `20261117_difusco_rbsmall_heatflip_lam0_s0` | 5–8，每卡 10 张 | 重新加噪 | bl9qbv4m |
| `20261117_difusco_rbsmall_randflip_s0` | 9，batch 40 | D3PM 后验 | 3zdeizwh |
| `20261116_difusco_rbsmall_heatflip_lam0p1_s0`（旧，patience 20） | 0，单卡 | 重新加噪 | c2ph9w5g |

- 新三组：`--patience 1000`（大于 200，不会早停），`--save_every_epoch`（每个 epoch 存 `epoch_XXXX.pt`）。启动脚本在 `experiments/20261117_difusco_rbsmall_{heatflip,randflip}/launch.sh`。
- 4 卡需要 `NCCL_IB_DISABLE=1`，否则 NCCL 探测 InfiniBand 时段错误（已写进启动脚本）。
- 预计：4 卡两组每个 epoch 约 3 分钟，约 10 小时跑完；单卡 randflip 约 15 小时以上。实际随机器负载变化（别的用户的 CPU 负载在 40–75 之间波动）。
- 旧 λ = 0.1 按用户要求不停。截至 11-17 约 10:55，跑到第 85 个 epoch，验证最好 19.92（第 84 个 epoch）。
- 提醒：11-17 11:41（匹兹堡时间）会检查一次三组新实验，报实测速度和 ETA。

## 6. 测速结论（11-17，λ = 0.1，负载约 40）

| 卡数 | 训练 100 步 | 验证 64 张 | 每个 epoch |
|---|---|---|---|
| 1 | 2:19 | 4:53 | 约 7.2 分钟 |
| 2 | 2:12 | 2:30 | 约 4.7 分钟 |
| 4 | 1:14 | 1:29 | 约 2.7 分钟 |

验证按卡数线性加速，训练主要受 CPU 限制。3 张卡凑不出每步 40 张，不用。

## 7. 下一步

1. 三组跑完后，各用 best.pt（以及需要的话，最后一个 epoch 的权重）在 1000 张 test 上测两个推理种子；randflip 用两种采样器都测，补全 2 × 2 表。用 `experiments/20261116_difusco_rbsmall_heatflip/test.sh` 的写法，改路径即可。
2. 比较：λ = 0.1 对 λ = 0（只差前向），λ = 0 对 randflip（只差采样器）。
3. 待用户定：SL 的 baseline 是否改成“DIFUSCO + 重新加噪”；之后补种子 1、2；阶段 2（后验对齐）什么时候开始。
4. 如果 λ = 0.1 明显更好，要加一个“同样制造冲突、但不看能量”的加噪对照，才能归因到地形。

## 8. 清理状态

- 11-16 中途停掉的几次运行，wandb 记录和本地文件都已删除。
- 测速临时输出在 `runs/scratch/speed_1794929955/`（不进 git，可删）。
