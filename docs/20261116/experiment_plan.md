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

  验证集也被分到 10 卡上：DiffUCO 用 64 张（ER 和 RB-small 都是），补齐到每卡 7 张，一共 70 张，所以分数都是 1/70 的倍数；DIFUSCO 的消融不限张数，RB-small 验证集 500 张，每卡 50 张。
- **等效 batch**：DiffUCO 在 RB-small 上每步 40 张图（4 × 10），在 ER 上每步 10 张；Shengyu 本地的 DIFUSCO 脚本每步 50 张（5 × 10）。我们单卡训练直接用 batch 40，两边都是每步 40 张图（11-16 定，见 §6.3）。
- **卡时**：前面的时间都是 10 卡的墙钟时间，换成卡时要乘 10。
- **表 1 的 DIFUSCO（RB-small 18.74，差 6.77%，5.33 分钟）在 sa 上找不到记录**（11-15 查）：
  - 查过的范围：`/home/shengyuf/CAM4CO/wandb` 全部 422 个 run，包括没有元数据的，也扫了 wandb 的二进制记录；`/usr1/shengyuf/RiemannCO/` 下没有 DIFUSCO 的输出目录；`CAM4CO/experiments/` 里只有 FM；全盘按文件名搜 `*difusco*` 也没有对应的产物。
  - 5 月出表期间，sa 上只跑过 CAM、DiffUCO、SDDS，一次 DIFUSCO 都没有。
  - sa 上的 DIFUSCO 只有 09-30 以后的 3 次 RB-small 50 epoch 消融，验证最好 17.9–18.9，都不是出表的配置。
  - 线索：Shengyu 11-07 的 `scripts/run_orig_sweep.sh` 里有 MIS RB-small 上 DIFUSCO × 3 个种子，用的是原 DIFUSCO 设置（总 batch 40、200 epoch、patience 20），日志写到 `logs/orig_sweep/`，但 sa 上没有这个目录。那几个提交是在 sf 集群（sf.lti.cs.cmu.edu）上做的，sweep 很可能也在 sf 上跑。表 1 的 DIFUSCO 要问 Shengyu 在哪台机器、用的什么配置。DiffUCO 出表的 RB-small run 在：annealing，05-13 21:46 开始，验证最好 20.27（第 137 个 epoch），test 19.87、19.88。完整配置见 `cam_paper_diffuco_runs.md` §3.1。

### 6.3 DIFUSCO 在 RB-small 上自己调；Shengyu 的启动脚本；表 1 的 DIFUSCO 可能用了错位的标签（11-15）

**用户决定**：RB-small 上的 DIFUSCO 由我们自己调参。

**Shengyu 目录里的三份 RB-small DIFUSCO 启动脚本**（`/home/shengyuf/CAM4CO/scripts/`，只读）：

| 脚本 | 时间 | 每卡 batch × 卡数 | 学习率 / weight decay | epoch / patience | 其他 |
|---|---|---|---|---|---|
| `mis/mis_rb_small_difusco.sh` | 内容最后改于 06-06（08-28 只是文件时间） | 5 × 8 = 40（git 里的 `config.sh`）；本地未提交的 `config.sh` 是 10 卡 = 50 | 2e-4 / 1e-4 | 200 / 20 | `num_tp` 1000，推理 cosine，50 步 × 20 次，验证 64 张；**没有 `--do_train`**，照原样运行只做 test。逐项核对见下 |
| `train_orig.sh`（由 `run_orig_sweep.sh` 调用，3 个种子） | 11-07 | 40 ÷ 卡数，总 batch 固定 40（注释：“原始 run 是 8 卡 × 5”） | 2e-4 / 1e-4 | 200 / 20 | 同上；种子可设 |
| `mis/rb_small_time_ablation.py` 等 | 09-30 起 | 5 × 10 | 2e-4 | 50 / 51 | 网络加时间输入的消融 |

**`mis/mis_rb_small_difusco.sh` 逐项核对（11-15）**：对照 RiemannCO 当前代码 `main.py` 的默认值和 `solvers/meta_solver.py` 的训练循环。

| 类别 | 参数 | 值 | 来源 |
|---|---|---|---|
| 网络 | `model` / `num_l` / `num_h` | EncodeProcessDecode / 8 / 64，输出 2 类 | 脚本 |
| 扩散 | 训练步数 `num_tp` / 加噪表 `diffusion_schedule` / `time_scale` | 1000 / linear / 1000 | 脚本 / 默认 / 默认 |
| 优化 | 优化器 / 学习率 / weight decay | AdamW / 2e-4 / 1e-4 | 默认 / 脚本 / 脚本 |
| | 学习率表 | cosine 衰减到 0，按“200 epoch × 每 epoch 步数”算 | 默认 |
| | 梯度裁剪 | 无 | 默认 |
| | 精度 | fp32（`--mixed_precision=no`） | 脚本 |
| 数据 | 训练集 | `RB_small_train` 全部 4000 张，每个 epoch 过一遍（SL 不看 `samples_per_epoch`） | 默认 |
| | 每卡 batch × 卡数 | 5 × `$PROCESSES`：git 里是 8（每步 40 张，每 epoch 100 步），本地是 10（每步 50 张，80 步） | 脚本 + `config.sh` |
| 训练长度 | `epochs` / `patience` | 最多 200 / 验证 20 个 epoch 不涨就停 | 脚本 |
| 验证 | `valid_sample`（argparse 前缀匹配到 `valid_samples`） | 64 张（8 卡正好 64；10 卡补成 70） | 脚本 |
| 推理 | `num_t` × `num_k` / 推理时间表 | 50 步 × 20 次 / cosine | 脚本 |
| 其他 | 种子 | 0 | 默认 |

- 学习率表用的是数据分到各卡之前的步数（每 epoch 800 步）。但 accelerate 每次更新会让学习率表走“卡数”步，两者正好抵消，所以学习率确实在 200 个 epoch 内衰减到 0，没有问题。
- **这份脚本不是出表时的训练命令**：
  - 它没有 `--do_train`，照原样跑只会读 `save_dir/best.pt` 做 test。
  - 它的内容是 06-06 的“clean”提交改的，那是 Shengyu 在笔记本上为公开代码整理脚本，在论文（arXiv 2605，5 月）之后。
  - git 里的 `config.sh` 在那次提交里是 `DATADIR=/mnt/task_runtime`、`PROCESSES=8`，看起来是另一个集群的路径。
- 更早的两个版本（04-02、04-18）是训练命令，带 `--do_train`：每卡 batch 8，梯度累积 2，最多 50 个 epoch，验证时只采 8 次（`num_k 8`），学习率同为 2e-4，推理 cosine。当时 `PROCESSES=8`，等效 batch 8 × 8 × 2 = 128。时间上它更可能接近表 1 那次，但没有运行记录能证实。

三份的核心设置相同：学习率 2e-4，weight decay 1e-4，最多 200 个 epoch，验证 20 个 epoch 不涨就停，训练 1000 步扩散，推理用 cosine 时间表 50 步 × 20 次。没有一份能对上表 1 那次 run。

**RB 标签错位的影响**（11-15 实测，`/usr1/shengyuf/MIS/RB_small_train` 随机抽 100 张）：
- 100 张图的节点插入顺序都和编号顺序不同。
- 用修复前的加载代码，96 张图的标签加载后不是独立集；修复后 0 张。标签平均大小 20.22。
- RiemannCO 里 `datasets/mis.py` 的修复在 11-07（提交 `f02c92f`），之前一直是错的。

**推论**：
- 11-07 之前在 RB 上训练的 SL 模型，基本都是用错位的标签训的。包括 Shengyu 09-30 的 DIFUSCO 消融（验证最好 17.9–18.9）。
- 如果表 1 的 DIFUSCO 也用的这套代码（附录 B.1 说所有扩散基线“在统一设置下重新实现和训练”），那 RB-small 上的 18.74 很可能也是错位标签训出来的。那样表 1 里 RB 上“UL 远好于 SL”的差距，有一部分可能来自这个 bug，而不是方法本身。
- ER 不受影响：ER 图的两种顺序一致，`slul_results.md` 里 ER 上的结论不受影响。
- UL 方法（DiffUCO、CAM、SDDS）不用标签，不受影响。
- 要找 Shengyu 确认：表 1 的 DIFUSCO 是哪天、用哪份代码训的。

**用户决定（11-16）**：DIFUSCO 每步 **40 张图**，和 DiffUCO 出表那次对齐。
- DiffUCO 那次核对过：每卡 4 张 × 10 卡，每张图 10 条轨迹。
  - `generate_data` 把全部轨迹打包成一个数据项返回，所以每步只更新一次参数。
  - 每次更新 = 40 张图、400 条轨迹。
- 我们单卡训练时直接用 batch 40。网络里的 GraphNorm 按单张图归一化，所以单卡 40 张和多卡分摊在数学上等价，只差随机性。
- 11-15 曾暂定 50 张（对应 Shengyu 本地 DIFUSCO 脚本的 5 × 10 卡），已改。

**我们调参的起点**：以上面三份脚本的共同设置为中心，学习率 2e-4、weight decay 1e-4、200 epoch、patience 20、每步 40 张图、cosine 推理。用修好的代码（`cam/UPSTREAM.md` 第 39、40 条）。具体网格待定。

### 6.4 11-16 00:05 UTC 的状态

- **ER 上的任务全部停掉**（用户 11-16）：给 16 万张图打标签的两个进程、UL 图缓存、DIFUSCO 1000 epoch × 3 个种子、旧的 DiffUCO × 6、评测队列。已经打好的标签和缓存留在磁盘上，没删。相关提醒已关闭。10 张卡全部空出来。另一个会话 2 天前开的 ER 评测队列（pid 3295845）没动。
- **DiffUCO 不再复现**（用户 11-16）：直接用 CAM 论文表 1 那次的记录，导入为 `experiments/20260513_diffuco_rbsmall_annealing/`（11-16 改名；同时导入了只换噪声的 `experiments/20260513_diffuco_rbsmall_bernoulli/`），训练曲线已同步到 wandb 项目 `yunyi-organization/SLVUL`。上面 §6.1 里“复现 + 多种子”的提议作废。
