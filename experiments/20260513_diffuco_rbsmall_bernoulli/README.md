# 20260513_diffuco_rbsmall_bernoulli：CAM 论文同批的 RB-[200-300] DiffUCO（bernoulli 噪声）

## 目的

和 `20260513_diffuco_rbsmall_annealing`（表 1 用的那次）配对：同一天、同一份代码、同样的超参，只换了加噪方式。用来看 DiffUCO 在 RB-small 上两种噪声的差别。表 1 选的是 annealing，这次没进表。

这是一次导入的实验：训练和测试都是 Shengyu 在 2026-05-13/14 跑的，这里只做记录和同步，不重跑（用户 11-16）。

## 配置

- 原始启动命令：`launch_original.sh`，取自 wandb 元数据。
- 和 annealing 那次的启动命令逐项比过，只有三处不同：`--noise_energy bernoulli`、`--save_dir`、`--wandb_run_name`。
- 其余全同：
  - 网络 8 层 × 64；训练 6 步、每张图 10 条轨迹；
  - β 1.2；τ 从 0.1 线性退到 0（1000 个 epoch 内）；
  - RAdam，学习率 1e-3，cosine 衰减，梯度裁剪 1；
  - 每 epoch 4000 张图，每次更新 40 张图（每卡 4 张 × 10 卡）；
  - 最多 1000 个 epoch，patience 200；验证 64 张（实际 70 张）；推理 50 步 × 20 次。
- bernoulli 噪声：中间每一步的“噪声能量”是向均匀分布靠拢的 Bernoulli 转移的负对数概率，翻转率 β_t = 0.5·2^(−6t)，乘以 τ，随 τ → 0 一起消失。annealing 噪声则是 t/T × 软能量，不随 τ 消失。见 `docs/20261116/slul_results.md` §5。
- 硬件：sa，10 张 RTX 2080 Ti 数据并行。代码：RiemannCO `ebae8b8`。
- 卡时：每 epoch 约 28 秒（10 卡）。到第 101 个 epoch 约 47 分钟，即约 8 卡时；到早停约 2.4 小时，即约 24 卡时。

## 状态

| run | wandb 原名 | 开始（UTC） | 结束 |
|---|---|---|---|
| 训练 `biczvw9h`（wandb 上已换成重放的 `bxfcarwm`） | diffuco_bernoulli | 2026-05-13 18:05 | 第 301 个 epoch 早停（约 20:27） |
| 测试 `w5tfv6zb` | test_diffuco_bernoulli | 2026-05-13 21:04 | 21:09 |
| 测试 `8hf0zjro` | test_diffuco_annealing（启动参数写的 annealing，但读的是 bernoulli 的权重） | 2026-05-14 00:47 | 00:47 |

同一天 17:56 和 17:57 还有两次没跑成的尝试：一次 test 没有结果，一次训练到第 13 个 epoch 就停了。这两次没有导入。

## 结果

| run | 验证最好（epoch） | test（1000 张） | wandb |
|---|---|---|---|
| 训练 | 20.11（101） | — | [20260513_diffuco_rbsmall_bernoulli](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/bxfcarwm) |
| 测试 05-13 | — | 19.701（每张 0.33 秒） | 没上传 wandb，本地记录在 `wandb/` |
| 测试 05-14 | — | 19.739（每张 0.36 秒） | 没上传 wandb，本地记录在 `wandb/` |

- 噪声类型只在训练时用到，测试不用。所以 05-14 那次虽然参数写的是 annealing，实际就是 bernoulli 权重的又一次测试。两次相差 0.04，来自推理采样的随机性。
- `runs.csv`：三个 run 的超参和结果（`cam/scripts/mis/collect_wandb_runs.py` 生成）。
- 权重：`runs/experiments/20260513_diffuco_rbsmall_bernoulli/best.pt`，复制自 `/usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_bernoulli/best.pt`，MD5 一致，不进 git。文件时间 05-13 18:53，等于开始时间加 101 个 epoch × 约 28 秒，所以是第 101 个 epoch 按验证集存下的。

## wandb 记录是怎么来的（11-16）

1. 从 `/home/shengyuf/CAM4CO/wandb/` 原样复制三个 run 目录到 `wandb/`。
2. 只把训练 run 用 `wandb sync --include-online --include-synced -p SLVUL -e yunyi-organization` 上传；302 个验证点，最好 20.114。测试 run 不上传，和 annealing 那边一致。
3. 改名为 `20260513_diffuco_rbsmall_bernoulli`，加标签 `cam_paper`、`imported`、`rb_small`、`DiffUCO`、`bernoulli`。
4. 补了正号曲线 `eval/mis_size`（横轴 `epoch_idx`）。summary 里写了：
   - `eval/mis_size_best` = 20.114；
   - `test/mis_size` = 19.701（05-13）；
   - `test/mis_size_0514` = 19.739。
5. 进 git 的 `config.yaml` 去掉了 wandb 自动加的 `_wandb` 段（含 Shengyu 的邮箱、主机和 GPU 序列号）。
6. 11-16 发现补的 `eval/mis_size` 接在第 3 万多步之后，和按当前代码跑的新实验（`eval/mis_size` 和 `eval/object` 在同一步，每个 epoch 100 步）对不上。用 `cam/scripts/mis/replay_wandb_run.py` 把训练 run 重放成新 run `bxfcarwm`：原有每一行按原步数写回，每个验证点在同一步加上 `eval/mis_size` = −`eval/object`，名字、超参、标签和 test 相关的 summary 照抄；不重放系统指标和原始时间戳。逐步核对 302 个点完全一致后，删掉了旧 run `biczvw9h`。本地 `wandb/` 里新旧两个目录都留着。

## 和 annealing 对比

| | 验证最好（epoch） | test |
|---|---|---|
| annealing | 20.27（137） | 19.872 / 19.876 |
| bernoulli | 20.11（101） | 19.701 / 19.739 |

- annealing 在验证集上高 0.16，在 test 上高 0.14–0.18。
- 只有一个种子，种子之间的波动还不知道。
- 两次选权重时温度都还没退完：bernoulli 在第 101 个 epoch，τ ≈ 0.090；annealing 在第 137 个 epoch，τ ≈ 0.086。都不是训到收敛的结果。
