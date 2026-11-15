# 第一步实验计划：SL 和 UL 在相同调参预算下的收敛表现（11-15 定）

目的：在同一个网络上，把 SL（DIFUSCO）和 UL（DiffUCO 两种噪声）都调好、训到收敛，得到一个稳定的“谁更好”的判断，再做后续的机制分析。此前 16k 张训练图上的任务都只算试跑，不进入这一步的结论。

## 1. 原则（用户 11-15 定）

1. **同一个网络**：EncodeProcessDecode，8 层 × 宽 64，不输入时间。
2. **同一个数据池**：163,840 张 ER-[700-800] 训练图，与 DIFUSCO 原文 ER 训练集同规模（附录 E）。SL 用 KaMIS 标签，UL 不用标签。验证集 100 张，test 集 128 张，都不变。
3. **调参组数相同**：三条线各 6 组，都是“学习率 + 一个方法特有的超参”。
4. **每组卡时对齐**：SL 每组的 GPU 卡时对齐 DiffUCO 每组（约 55 卡时，2080 Ti，单卡独占时测）。SL 补算力时，同时扩大数据量，不是只加 epoch。
5. **只在验证集上挑**：超参和检查点都按验证集选。test 集只在第 4 阶段评一次。
6. **收敛**：最终配置把训练拉长到 3 倍，主指标涨幅小于种子间的波动，才算收敛。
7. **报告**：按 Dodge et al. 2019 的做法画两条曲线，横轴分别是试验次数和卡时。两种口径结论一致，才算稳定。SL 标签的求解器时间单独列出。

## 2. 三条线的调参网格

| 线 | 超参 1 | 超参 2 | 固定的超参 |
|---|---|---|---|
| SL：DIFUSCO | 学习率 2e-4（原文）、5e-4（CAM 统一）、1e-3 | 训练噪声表 linear（原文）、cosine | AdamW，weight decay 1e-4，batch 32，T=1000，epoch 数 E_SL 由测速定（约 350） |
| UL：DiffUCO annealing | 学习率 5e-4（CAM 统一）、2e-3（原文表 7） | 初始温度 0.1（CAM 统一）、0.2、0.4（原文 0.3–0.4） | RAdam，梯度裁剪 1.0，100 epoch × 1000 张图 × 20 条轨迹 × 50 步，温度线性退到 0 |
| UL：DiffUCO bernoulli | 同上 | 同上 | 同上 |

- 推理：三条线都是 50 步 × 20 次采样。
- 噪声类型不算调参维度。两种噪声各自和 SL 比；若报“两种噪声里最好的 UL”，SL 也要给 12 组。

## 3. 阶段和任务

卡时以 2080 Ti 单卡独占计。DIFUSCO 只占约 1 GB 显存，可以和别的任务共用一张卡，但预算按独占时的速度算。

### 阶段 0：数据和测速（CPU 为主）

| 编号 | 内容 | 资源 | 状态 |
|---|---|---|---|
| D1 | 生成 147,456 张新图，编号 16384–163839，生成器与现有图一致（`cam/scripts/mis/gen_er700_graphs.py`） | CPU 16 核，约 1 小时 | 11-15 完成。另补生成 100 张（编号 163840–163939）：旧图编号从 100 开始只有 16,284 张，补齐后共 163,840 张。训练池是 `/home/shengyuf/yunyi/data/slul_er700/er_train`（全是软链接） |
| D2 | 编译 KaMIS（基准框架锁定的 commit 791334e），在 `yunyi/tools/KaMIS/deploy` | `yunyi/envs/buildtools` 里的 cmake 4，需要加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` | 11-15 完成 |
| D3 | 定 KaMIS 时间上限：DIFUSCO 的数据说明（`DIFUSCO-LNS/data/README.md`）用的是 `--time_limit 60`，照用。在 40 张已有标签的图上，60 秒的新标签平均 44.68，旧标签 44.35，有 21 张一样大。为了标签质量一致，**163,840 张全部用 60 秒重新打标签**，旧标签不用 | CPU 约 1 小时 | 11-15 完成 |
| D4 | 给 163,840 张图打标签（`cam/scripts/mis/label_kamis.py`，每张 60 秒） | 约 2,730 CPU 小时；机器负载全是我们自己的任务，用 32–64 核约 2–3.5 天。时间上限按墙钟算，抢 CPU 会降低标签质量，所以要盯着平均大小 | 11-15 开始。两批：旧的 16k 张 32 进程，新图 24 进程；旧的打完后新图加到 56 进程。17:47 UTC 读数：平均大小 44.78 / 44.75，每张 61–62 秒，负载约 81–87。预计 11-17 晚上打完 |
| D5 | 组装训练池（已完成）；建图缓存。**缓存里存着建缓存时读到的标签，标签缺失时存的是“无标签”**，所以分两套：UL 用 `.cache/cam/graphs_ul_pool`（UL 不读标签），11-15 17:49 开始建，约 3.3 张/秒，预计 11-16 07:45 建完；SL 用 `.cache/cam/graphs_sl_pool`，等全部标签打完再建（`runs/queue_slul_er700/launch_pool.sh` 会检查标签是否齐全） | CPU 12 核 | UL 缓存进行中 |
| D6 | 测速：SL 和 DiffUCO 在新池上各跑 2 个 epoch（单卡独占），定 E_SL，使 SL 单组卡时约等于 DiffUCO 单组 | GPU 约 2 小时 | D5 之后 |

- UL 不需要标签，UL 缓存建完就能开 UL 的粗筛，不用等 D4。启动脚本 `runs/queue_slul_er700/launch_pool.sh`，任务名以 `pool_` 开头，评测队列会照常评测。

### 阶段 1：粗筛（每组 1 个种子，种子 0）

| 编号 | 线 | 学习率 | 超参 2 | 卡时 |
|---|---|---|---|---|
| S1 | SL | 2e-4 | linear | 约 55 |
| S2 | SL | 5e-4 | linear | 约 55 |
| S3 | SL | 1e-3 | linear | 约 55 |
| S4 | SL | 2e-4 | cosine | 约 55 |
| S5 | SL | 5e-4 | cosine | 约 55 |
| S6 | SL | 1e-3 | cosine | 约 55 |
| A1 | UL annealing | 5e-4 | 温度 0.1 | 约 55 |
| A2 | UL annealing | 5e-4 | 温度 0.2 | 约 55 |
| A3 | UL annealing | 5e-4 | 温度 0.4 | 约 55 |
| A4 | UL annealing | 2e-3 | 温度 0.1 | 约 55 |
| A5 | UL annealing | 2e-3 | 温度 0.2 | 约 55 |
| A6 | UL annealing | 2e-3 | 温度 0.4 | 约 55 |
| B1–B6 | UL bernoulli | 同 A1–A6 | 同 A1–A6 | 各约 55 |

- 合计 18 组，约 990 卡时。
- 每组每个 epoch 都在验证集上评一次，按验证集取 `best.pt`，每组选出前 2 名。

### 阶段 2：复核（前 2 名补种子）

- 每条线前 2 名各补种子 1、2，共 12 组，约 660 卡时。
- 按 3 个种子的验证集均值，每条线选出最终配置。

### 阶段 3：收敛检查

- 每条线的最终配置把训练拉长到 3 倍（SL 3·E_SL epoch，UL 300 epoch），种子 0，共 3 组，约 495 卡时。
- 单组墙钟约 7 天，是整个计划的关键路径。
- 3 倍长度的主指标涨幅小于种子间的波动，就算收敛；否则继续加长，或用更长的那档作为最终配置。

### 阶段 4：最终评测和报告

- 每条线的最终配置用 3 个种子在 test 集上评一次，报均值、标准差，以及逐图的配对比较。
- 画两条曲线：最好验证值随试验次数的变化，以及随卡时的变化（Dodge et al. 2019）。
- 单独列出：SL 标签的 KaMIS 求解时间。

### 合计

| 阶段 | 组数 | 卡时 |
|---|---|---|
| 1 粗筛 | 18 | 约 990 |
| 2 复核 | 12 | 约 660 |
| 3 收敛 | 3 | 约 495 |
| 合计 | 33 | 约 2,150 |

- 10 张 2080 Ti 全用上约 9–10 天。加上阶段 3 的串行等待，总共约 2.5–3 周。
- 拿到新 GPU 后，卡时要按新卡的实测速度重算，并且两边用同一种卡，或者都换算成同一种卡的时间。

## 4. 风险

- **SL 的 epoch 数**：16 万张图训约 350 epoch，每张图会被看约 350 次，比 DIFUSCO 原文的 50 次多。如果 SL 在阶段 1 的验证曲线早就平了，说明多出的算力没用上，可以考虑再扩大数据（32 万张，约 175 epoch），代价是标签时间翻倍。
- **标签质量**：KaMIS 时间上限会影响标签质量，这是 H1（标签质量）的变量。D3 的标定要做扎实，新旧标签的平均大小要对得上。
- **CPU 紧张**：机器负载已有约 75/80，D4 可能要好几天。如果新 GPU 机器有空闲 CPU，可以把打标签挪过去。
- **UL 的数据池变大**：DiffUCO 每个 epoch 只抽 1000 张图，100 epoch 只看 10 万次，数据池从 16k 扩到 16 万后，每张图大多只被看一次以内。这和旧设置不同，所以旧的 DiffUCO 结果不能直接拿来比。

## 5. 旧任务的处理（用户 11-15 同意）

- `DIFUSCO_e3000_s0`（卡 1）：11-15 16:45 UTC 已停（用的是旧的 16k 数据池），也已从评测队列中移除。
- DIFUSCO 学习率试跑（卡 8、9）和 1000 epoch 档（卡 1、8、9）：跑完为止，结果留作参考。
- DiffUCO 100 epoch 的 6 个旧任务（卡 2–7）：等新数据池建好（D5）、测完速（D6）就停，把卡让给阶段 1 的 UL 任务。

## 6. CAM 论文的统一设置和 Shengyu 的脚本（11-15 查）

用户转述 Shengyu 的话：CAM 论文里小 GNN 上的 DIFUSCO 和 DiffUCO 是用这个集群跑的，已经调过参。

**附录 B.2 写了的**（`ref/lit_pdfs/2605.30920.txt`）：
- 训练图 16,284 张，KaMIS 标签。
- 网络 8 层 × 64，不输入时间步。
- 优化器：DIFUSCO 用 AdamW；DiffUCO 用 RAdam，梯度裁剪 1.0。weight decay 1e-4。
- 学习率“从 [1e-4, 1e-3] 中选”，温度 τ0 ∈ {0.1, 0.05}，线性退火到 0。
- 推理 50 步 × 20 次。
- 噪声类型在 bernoulli 和 annealing 中取好的那个（“大多是 annealing”）。

**附录没写的**：epoch 数、试了多少组、按验证集还是 test 集选、Table 1 每个方法最终用的哪组超参。

**Shengyu 的脚本**（`/home/shengyuf/CAM4CO/scripts/mis/mis_er_difusco.sh`、`mis_er_diffuco.sh`，2026-08-28）和我们复现的差别：

| | Shengyu 的脚本 | 我们的复现（`slul_er700.sh`） |
|---|---|---|
| DIFUSCO 学习率 | 2e-4 | 5e-4 |
| DIFUSCO batch | 每卡 5，10 个进程（总 50；`train_orig.sh` 注释写总 batch 40 = 8 卡 × 5） | 32，单卡 |
| DIFUSCO epoch | 200，patience 20（会早停） | 100 / 1000，不早停 |
| DiffUCO 学习率 | 1e-3 | 5e-4 |
| DiffUCO 训练轨迹步数 `num_tp` | **12** | **50** |
| DiffUCO 冲突惩罚 β | **1.2** | 1.02（默认） |
| DiffUCO 噪声 | bernoulli（脚本里） | bernoulli 和 annealing 都跑 |
| DiffUCO epoch | 1000（温度在 1000 epoch 内退完） | 100 |
| 硬件 | `PROCESSES=10`（10 卡数据并行；sa 正好 10 张 2080 Ti），`DATADIR=/usr1/shengyuf` 在 sa 上存在 | 2080 Ti 单卡 |

**影响**：
- DiffUCO 训练轨迹从 50 步改成 12 步，每个 epoch 的计算量约降到 1/4，§3 的卡时估计要重算。12 步也更接近 DiffUCO 原文的做法：SDDS 论文附录 A.10 提到 DiffUCO 原文只用 3–6 步，SDDS 用到 12 步。
- β 从 1.02 改成 1.2 会改变能量地形，属于 UL 的超参。
- 脚本未必就是 Table 1 的最终配置：脚本里写的是 bernoulli，附录却说大多用 annealing。DIFUSCO 脚本的 test 路径指向 `er_large_test`。

**更正（11-15）**：之前写的“A6000”没有依据。CAM 论文正文和附录都没提 GPU 型号。`config.sh` 里只有一行没人引用的 Slurm 默认参数 `--gres=gpu:A6000:8`，而且它和同文件里的 `PROCESSES=10` 也对不上。`handoff.md` 里“CAM 论文用的是 48 GB 的 A6000”同样没有出处。数据目录在 sa 上，10 个进程也正好对应 sa 的 10 张卡，和 Shengyu 说的“用这个集群跑的”一致。

**sa 上的实际产物和运行记录（11-15 补，用户已和 Shengyu 确认）**：表 1 对应的运行记录在 `/home/shengyuf/CAM4CO/wandb`（Shengyu 的私有仓库 RiemannCO）。DiffUCO 的全部 run 整理在 `cam_paper_diffuco_runs.md` 和 `data/cam_paper_diffuco_runs.csv`。要点：
- 表 1 的 43.98 = annealing、`num_tp=6`、学习率 1e-3、τ0 0.1、β 1.2，权重是 `/usr1/shengyuf/RiemannCO/mis_er_diffuco/diffuco_annealing/best.pt`（按验证集选在第 141 个 epoch，第 341 个 epoch 早停）。
- 上面的脚本（bernoulli、`num_tp=12`）不是出表的配置。
- `diffuco_bernoulli/best.pt` 已被 `num_tp=6` 那次覆盖，不是验证集最好的 bernoulli 权重。
- 没找到 DIFUSCO 的权重。

**向 Shengyu 确认的结果**：
1. 表 1 的 DiffUCO 用的哪组超参：已从运行记录查清（见上）。DIFUSCO 的还没查。
2. 调参怎么调的：本地记录里 ER 上只变过噪声类型和 `num_tp`（6、12），学习率、τ0、β 都没变。两种噪声在验证集上打平（44.20 对 44.19），最后按 test 选了 annealing。
3. 调参记录：就是上面的 wandb 本地目录。

### 6.1 对计划的影响（11-15，待和用户一起定）

- **DiffUCO 不再调参**（用户和 Shengyu 确认）。§2 里 DiffUCO 的 6 组调参和阶段 1、2 里对应的任务取消，改为：
  1. 用我们的评测流程把 `diffuco_annealing/best.pt` 在 128 张 test 图上测几次，确认能复现 43.98 左右。这一步也检验两边评测是否一致。
  2. 按表 1 的配置（annealing、`num_tp=6`、学习率 1e-3、τ0 0.1、β 1.2、每 epoch 1000 张图、20 条轨迹、patience 200）用我们的代码换 2–3 个种子重训，量出种子之间的差异。
- **要一起讨论的口径问题**：
  - 这个 checkpoint 是早停选出来的，温度只退到约 0.086，不是“训到收敛”。按 `slul_results.md` §3 的收敛口径，要不要再补一次训满 1000 个 epoch 的？
  - 两种噪声的取舍是按 test 定的。SL 那边如果严格按验证集选，两边口径不一致；差距约 0.1，在种子差异的量级内。
  - SL 和 UL 的算力对齐要重算：DiffUCO 出表的那次到最好的 checkpoint 约 26 卡时、跑到早停约 63 卡时（10 卡 × 约 66 秒每 epoch）。
- **我们之前的 DiffUCO run**（`num_tp=50`、β 1.02、学习率 5e-4、100 个 epoch，每 epoch 同样 1000 张图）和表 1 的配置差得较远，只作参考。

**另外**：`/home/shengyuf/CAM4CO/scripts/config.sh` 里以明文写着一个 wandb API key，我们没用，也没复制，建议提醒 Shengyu 处理。

### 6.2 改到 RB-[200-300] 上做；当时每次实验用了几张卡（11-15）

**用户决定**：SL 和 UL 对比的验证先全部放在 RB-[200-300]（RB-small）上做。原因：
- 图小：200–300 个节点、3–4 千条边。ER-[700-800] 有 4–4.5 万条边。
- 按每张图算，训练快约 9 倍（DiffUCO 约 7 ms 对 66 ms）。
- CAM 表 1 里 UL 在这里领先得更明显：DIFUSCO 18.74（差 6.77%），DiffUCO 19.88（差 1.09%）。

上面各节的网格、卡时和阶段安排都是按 ER 写的，要按 RB-small 重排。注意：RB 图的节点插入顺序和编号顺序不一致，11-15 之前加载出来的 RB 标签全是错位的，现在已修好（`cam/UPSTREAM.md` 第 39 条）。所以 RB 上以前所有用到标签的 SL 结果都要重跑。

**每次实验都是单机 10 卡（10 张 2080 Ti，即 sa 整台机器）的数据并行**：
- 启动脚本：`accelerate launch --num_processes=$PROCESSES`，`config.sh` 里 `PROCESSES=10`。
  - 两份脚本都比 5 月的 run 晚：脚本是 08-28 的，`config.sh` 11-07 改过。所以下面又用日志核对了一遍。
  - Shengyu 后来的 RB-small 消融脚本（`rb_small_time_ablation.py`）里明写了 `--num_processes 10`，`CUDA_VISIBLE_DEVICES=0,...,9`。
- 日志核对：RiemannCO 每个 epoch 的步数是 `samples_per_epoch // 进程数 // batch_size`。

  | 方法 | 数据集 | 每 epoch 图数 | 每卡 batch | 日志里每 epoch 步数 | 推出的进程数 |
  |---|---|---|---|---|---|
  | DiffUCO | RB-small | 4000 | 4 | 100 | 10 |
  | DiffUCO | ER | 1000 | 1 | 100 | 10 |
  | DIFUSCO（09-30 起的消融） | RB-small | 4000（整个训练集） | 5 | 80 | 10 |

  验证集也被分到 10 卡上：ER 是 64 张，补齐到每卡 7 张，一共 70 张，所以分数都是 1/70 的倍数；RB-small 是 500 张，每卡 50 张。
- **等效 batch**：DiffUCO 在 RB-small 上每步 40 张图（4 × 10），在 ER 上每步 10 张；DIFUSCO 每步 50 张（5 × 10）。我们单卡复现时，要么用梯度累积补到同样的等效 batch，要么把 batch 当成一个要说明的差异。
- **卡时**：前面的时间都是 10 卡的墙钟时间，换成卡时要乘 10。
- 本地 wandb 里没找到 RB-small 上出表 1 的 DIFUSCO run，只有 09-30 以后的 50 epoch 消融（验证最好 17.9–18.9）。DiffUCO 出表的 RB-small run 在：annealing，05-13 21:46 开始，验证最好 20.27（第 137 个 epoch），test 19.87、19.88。
