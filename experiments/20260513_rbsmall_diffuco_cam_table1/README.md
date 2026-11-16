# 20260513_rbsmall_diffuco_cam_table1：CAM 论文表 1 里 RB-[200-300] 的 DiffUCO

## 目的

SL vs UL 对比里 UL（DiffUCO）这一侧的基准。用户和 Shengyu 确认（11-16）：DiffUCO 已调过参，表 1 报的就是最好的配置，我们**不再重跑**，直接导入这次的记录。

这是一次导入的实验：训练和测试都是 Shengyu 在 2026-05 跑的，这里只做记录和同步。

## 配置

- 原始启动命令：`launch_original.sh`（取自 wandb 元数据，只作记录，不在这里运行）。
- 主要超参：
  - 网络 8 层 × 64；
  - DiffUCO，annealing 噪声，训练 6 步、每张图 10 条轨迹；
  - β 1.2，τ 从 0.1 线性退到 0（1000 个 epoch 内）；
  - RAdam，学习率 1e-3，cosine 衰减，梯度裁剪 1；
  - 每 epoch 4000 张图（整个训练集），每次更新 40 张图（每卡 4 张 × 10 卡）；
  - 最多 1000 个 epoch，验证 200 个 epoch 不涨就停；验证 64 张（实际 70 张）；
  - 推理 50 步 × 20 次。
- 逐项说明见 `docs/20261116/cam_paper_diffuco_runs.md` §3.1。
- 硬件：sa，10 张 RTX 2080 Ti 数据并行。代码：RiemannCO `ebae8b8`（训练）、`7b03f86`（第二次测试）。
- 卡时：每 epoch 约 28 秒（10 卡）。到第 137 个 epoch 约 11 卡时，到早停约 26 卡时。

## 状态

| run | wandb 原名 | 开始（UTC） | 结束 |
|---|---|---|---|
| 训练 `gcgve1vl` | diffuco_annealing | 2026-05-13 21:46 | 第 337 个 epoch 早停（约 00:25） |
| 测试 `kur6gnkv` | test_diffuco_annealing | 2026-05-14 00:30 | 00:31 |
| 测试 `40tvlj2t` | test_diffuco_annealing | 2026-05-22 15:27 | 15:33 |

## 结果

| run | 验证最好（epoch） | test（1000 张） | wandb |
|---|---|---|---|
| 训练 | 20.27（137） | — | [cam_table1_rbsmall_diffuco_train](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/gcgve1vl) |
| 测试 05-14 | — | 19.872（每张 0.36 秒） | [cam_table1_rbsmall_diffuco_test_0514](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/kur6gnkv) |
| 测试 05-22 | — | 19.876（每张 0.37 秒）= 表 1 的 19.88 | [cam_table1_rbsmall_diffuco_test_0522](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/40tvlj2t) |

- `runs.csv`：三个 run 的超参和结果（`cam/scripts/mis/collect_wandb_runs.py` 生成）。
- 权重：`runs/experiments/20260513_rbsmall_diffuco_cam_table1/best.pt`（复制自 `/usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_annealing/best.pt`，MD5 一致，不进 git）。是训练第 137 个 epoch 按验证集存下的。

## wandb 记录是怎么来的（11-16）

1. 从 `/home/shengyuf/CAM4CO/wandb/` 原样复制三个 run 目录到 `wandb/`。
2. 用 `wandb sync --include-online --include-synced -p SLVUL -e yunyi-organization` 上传到我们的项目。训练 run 的曲线完整：338 个验证点，最好 20.271。
3. 在 wandb 上改了名字（原来两个测试 run 同名），加了标签 `cam_paper`、`imported`、`rb_small`、`DiffUCO`。
4. 两个测试 run 原本只在控制台输出结果，补写进 summary：`test/mis_size`、`test/time_per_graph_s`。
5. 11-16 补了一条正号的曲线 `eval/mis_size`（验证集 MIS 大小，= −`eval/object`），横轴用 `epoch_idx`（原始记录的步数已用完，只能接在后面写）。summary 里加了 `eval/mis_size_best` = 20.271。
6. 进 git 的 `config.yaml` 去掉了 wandb 自动加的 `_wandb` 段（含 Shengyu 的邮箱、主机和 GPU 序列号），超参一项不少；原件在 Shengyu 的目录和 wandb 上。

## 结论和注意

- 这是 RB-small 上 UL 的参照：验证 20.27，test 19.87–19.88。
- 选权重时温度只退到约 0.086，学习率也还在 cosine 衰减前段，不是训到收敛。
- 测试只用了种子 0，没有多种子的波动。
