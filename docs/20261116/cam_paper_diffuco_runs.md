# CAM 论文里 DiffUCO 的调参记录（11-15 整理）

用户和 Shengyu 确认：CAM 论文表 1 的结果对应的运行记录，就是 Shengyu 在 sa 集群上的 wandb 本地目录 `/home/shengyuf/CAM4CO/wandb`。代码仓库是 Shengyu 的私有开发仓库 RiemannCO（公开的 CAM 仓库 2026-06-01 才从它整理出来）。Shengyu 说 DiffUCO 已经调过参，论文报的就是最好的那组，我们不再重复调。

- 全部 DiffUCO run 的表（38 个，含 RB、BA 上的）：`data/cam_paper_diffuco_runs.csv`。
- 生成脚本：`cam/scripts/mis/collect_wandb_runs.py`（只读 wandb 目录）。超参以每个 run 的启动命令（`wandb-metadata.json` 里的 `args`）为准，`config.yaml` 只用来补默认值。验证集分数取自 `output.log` 每个 epoch 的 `eval/object`，test 分数取自最后的 `Avg. obj`。
- 注意：`config.yaml` 里的 `method` 字段不可靠。几个 run 的 `config.yaml` 写的是 DiffUCO，启动命令其实是 SDDS（例如 05-17、05-21 那几个 `num_tp=12` 的 test）。11-15 第一次口头汇报时把它们算成了 DiffUCO，这里已更正。

## 1. ER-[700-800] 上跑过的 DiffUCO

所有 run 共同的设置（启动命令里写明的）：
- 学习率 1e-3，cosine 衰减；RAdam，梯度裁剪 1。
- τ0 = 0.1，在 `N_anneal` = 1000 个 epoch 内线性退到 0。β = 1.2。
- 网络 8 层 × 64。
- 每个 epoch 1000 张图（`samples_per_epoch 1000`，不是整个 16,284 张的训练集）。每卡 batch 1，每个 epoch 100 步。所以每步 10 张图，和 10 个进程的数据并行对得上。
- 每张图采 20 条训练轨迹（`num_kp 20`）。
- 最多 1000 个 epoch，验证集 200 个 epoch 不涨就早停（`patience 200`）。
- 验证集：`er_valid`，分数是 70 张图的平均（每个值都是 1/70 的整数倍）。
- 推理：50 步 × 20 次；大图 ER-[9000-11000] 用 200 步。

能变的只有两个：噪声类型、训练轨迹步数 `num_tp`。

| 训练 run（开始时间，UTC） | 噪声 | `num_tp` | 验证最好（epoch） | 结束于 epoch | 在 test（128 张）上 |
|---|---|---|---|---|---|
| 05-15 03:39 | bernoulli | 12 | **44.20**（869） | 950（被停掉，未早停） | 43.88（05-16 01:37 测 1 次） |
| 05-16 01:37 | bernoulli | 6 | 43.49（52） | 252（早停） | 43.31、43.27（05-16 17:09、17:10） |
| 05-21 07:47 | annealing | 6 | 44.17（75） | 169（被停掉） | 没单独测 |
| 05-21 16:06 | annealing | 6 | **44.19**（141） | 341（早停） | **43.92、43.98、43.96、43.98** |

论文表 1：DiffUCO 在 ER-[700-800] 上 **43.98** = 最后一行，annealing、`num_tp=6`，checkpoint 是这个 run 在第 141 个 epoch 按验证集存下的 `best.pt`。

另外几个 test run 没有结果（05-16 15:48–15:53 的 5 次报错退出，05-22 00:23 没有日志）。ER-[9000-11000]：annealing 测到 372.75（200 步），bernoulli（`num_tp=6`）测到 354.31（50 步）；论文写的是 373.31，本地日志里没有这个数。

## 2. 读这些记录时要注意的

1. **验证集上两种噪声打平**：bernoulli（`num_tp=12`）44.20，annealing（`num_tp=6`）44.19。论文选 annealing，是因为它在 test 上更好（43.98 对 43.88）。所以“报两种里好的那个”实际是按 test 选的。差距约 0.1，和种子之间的差异同一量级。
2. **同一个 checkpoint 在 test 上测了 4 次**（43.92–43.98），论文报的是最高的。推理有随机性。
3. **磁盘上的 bernoulli 权重不是最好的那个**：
   - 两个 bernoulli 训练写进同一个目录（当时叫 `mis_er_diffuco/diffuco`，后来改名为 `diffuco_bernoulli`）。
   - `num_tp=6` 那次在 05-16 01:37 开始，覆盖了 `num_tp=12` 那次的 `best.pt`。
   - 现在的 `diffuco_bernoulli/best.pt` 修改时间是 05-16 02:36，正好是 `num_tp=6` 那次第 52 个 epoch（验证 43.49）。
   - 所以验证集最好的 bernoulli 权重（44.20）已经不在了。
4. **annealing 的 `best.pt` 没有训完温度退火**：
   - 它在第 141 个 epoch 存下，第 341 个 epoch 早停。
   - 温度按 1000 个 epoch 线性退火，第 141 个 epoch 时 τ ≈ 0.1 × (1 − 141/1000) ≈ 0.086，还远没到 0。学习率的 cosine 衰减也按 1000 个 epoch 算。
   - 也就是说，这是“早停选出来的”，不是“训到收敛”。按我们 §3 的收敛口径，它是否已经收敛要另外看。
5. **调的范围很窄**：学习率只用过 1e-3，τ0 只用过 0.1，β 只用过 1.2。附录 B.2 写的“学习率从 [1e-4, 1e-3] 中选、τ0 ∈ {0.1, 0.05}”，在这些记录里没有对应的 run。
6. **算力**：以 annealing 那次为例，每个 epoch 约 66 秒（10 卡）。到第 141 个 epoch 约 2.6 小时 × 10 卡 ≈ 26 卡时，跑到早停约 63 卡时。
7. **代码**：这些 run 用的是 RiemannCO 的提交 `ebae8b8`（训练）和 `7b03f86`（测试）。和我们拷来的公开 CAM 代码比，`solvers/diffuco.py`（DiffUCO 算法）、`envs/mis.py`（能量）完全相同；`utils/noise_distributions.py` 只差一个空行；网络文件只差 import 名和一个 DiffUCO 用不到的时间嵌入分支。`solvers/meta_solver.py` 差 123 行，主要是多卡训练的按大小分桶采样，还没逐行核对完。

## 3. 其他数据集（只列结果，细节见 CSV）

| 数据集 | 噪声 | 验证最好 | test | 论文表 1 |
|---|---|---|---|---|
| RB-[200-300] | bernoulli | 20.11 | 19.70；同一权重用 annealing 设定推理 19.74 | — |
| RB-[200-300] | annealing | 20.27 | 19.87、19.88 | **19.88**（annealing） |
| RB-[200-300]，带时间输入 | bernoulli / annealing | 18.96 / 20.11 | — / 19.74 | — |
| RB-[800-1200]（6 层） | bernoulli | 36.26 | — | |
| RB-[800-1200]（6 层） | annealing | 40.40 | 40.50 | 40.52（本地日志里没有这个数） |

BA 图上的 3 个 run 是最小支配集（MDS），不是 MIS。

### 3.1 RB-[200-300] 上出表 1 的那次 DiffUCO：完整配置

**结果**：论文表 1 RB-[200-300] 的 DiffUCO 19.88 就是这次。
- 训练 run：`run-20260513_214606-gcgve1vl`，05-13 21:46（UTC）开始，代码是 RiemannCO 的 `ebae8b8`，sa 上 10 张 2080 Ti 做数据并行。
- 验证集最好 20.27，在第 137 个 epoch；第 337 个 epoch 早停（137 + patience 200）。
- 权重：`/usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_annealing/best.pt`。文件修改时间 05-13 22:51，等于开始时间加 137 个 epoch × 每 epoch 约 28 秒，和第 137 个 epoch 对得上。
- test（1000 张图）测过两次，都用的这个权重，启动参数和训练相同，只是不加 `--do_train`：
  - 05-14 00:30，代码 `ebae8b8`：19.872；
  - 05-22 15:27，代码 `7b03f86`：19.876（四舍五入为论文的 19.88）。

**启动命令**（来自 `wandb-metadata.json`，原样照抄；用 `accelerate launch --num_processes 10` 启动）：

```
main.py --train_path /usr1/shengyuf/MIS/RB_small_train --valid_path /usr1/shengyuf/MIS/RB_small_valid
  --test_path /usr1/shengyuf/MIS/RB_small_test --problem mis
  --samples_per_epoch 4000 --valid_sample 64 --batch_size 4 --mini_batch_size 1
  --num_t 50 --num_k 20 --num_tp 6 --num_kp 10 --num_h 64 --num_l 8
  --tau0 0.1 --method DiffUCO --model EncodeProcessDecode --do_train --noise_energy annealing
  --optimizer RAdam --grad_clip 1 --beta 1.2 --lr 0.001 --epochs 1000 --N_anneal 1000
  --patience 200 --lr_scheduler cosine-decay
  --save_dir /usr1/shengyuf/RiemannCO/mis_rb_small_diffuco/diffuco_annealing
  --wandb_name mis_rb_small_diffuco --wandb_run_name diffuco_annealing
```

**逐项含义**：

| 类别 | 参数 | 值 |
|---|---|---|
| 网络 | `model` / 层数 `num_l` / 宽度 `num_h` | EncodeProcessDecode / 8 / 64，不输入时间 |
| 训练数据 | 训练集 | `RB_small_train`，4000 张 |
| | 每 epoch 图数 `samples_per_epoch` | 4000（正好一整轮） |
| | 每卡 batch `batch_size` × 进程数 | 4 × 10 = 每步 40 张图，每 epoch 100 步 |
| | `mini_batch_size` | 1 |
| 扩散 | 训练轨迹步数 `num_tp` | 6 |
| | 每张图的训练轨迹数 `num_kp` | 10 |
| | 噪声 `noise_energy` | annealing |
| | `diffusion_schedule` | linear（默认值） |
| 能量 | 冲突惩罚 `beta` | 1.2 |
| 温度 | `tau0` → `tau1`，`temp_schedule` | 0.1 → 0，线性，在 `N_anneal` = 1000 个 epoch 内退完 |
| 优化 | 优化器 / 学习率 / 衰减 | RAdam / 1e-3 / cosine，按 1000 个 epoch 算 |
| | 梯度裁剪 / weight decay | 1 / 1e-4（默认值） |
| | 精度 | fp32（脚本里 `--mixed_precision=no`） |
| 训练长度 | `epochs` / `patience` | 最多 1000 / 验证 200 个 epoch 不涨就早停 |
| 验证 | `valid_sample` | 64 张，分到 10 卡补齐成每卡 7 张，实际 70 张 |
| 推理 | 步数 `num_t` × 次数 `num_k` | 50 × 20，`inference_schedule` linear（默认值） |
| 随机种子 | `seed` | 0（默认值） |

“默认值”一项取自这次 run 的 `config.yaml`（argparse 的实际取值），启动命令里没写。

**时间和卡时**：每 epoch 约 28 秒（10 卡，含验证）。到第 137 个 epoch 约 64 分钟，即约 11 卡时；跑到早停约 2.6 小时，即约 26 卡时。

**要注意**：
- 选权重时温度只退到 τ ≈ 0.1 × (1 − 137/1000) ≈ 0.086，学习率也还在 cosine 衰减的前段，不是训到收敛。
- 08-28 的脚本 `scripts/mis/mis_rb_small_diffuco.sh` 和这次不同：bernoulli、`num_tp` 12、`num_kp` 20、每 epoch 1000 张。复现以上面的启动命令为准。
- 我们单卡复现时，每步 40 张图要用梯度累积补齐（每卡 4 张 × 累积 10 步），否则等效 batch 不同。

## 4. 对实验计划的影响

见 `experiment_plan.md` §6.1。
