# 20261118_fm_rblarge_mcmcstart：从 MCMC 起点出发的 flow matching / bridge matching（RB-[800-1200]）

## 动机（11-17 和用户商定）

- RB-small 上，MCMC 自己就能到 KaMIS：RLSA 3000 步 × 200 条链，每张图 1.5 秒，64 张验证图全部和 KaMIS 打平。所以“MCMC + 学习”在 RB-small 上测不出东西。
- RB-large 上 MCMC 到不了 KaMIS。RLSA 随预算的曲线在约 42 处变平（`runs/scratch/mcmc_budget_rblarge_1118/`，64 张验证图）：

  | 步数 × 链数 | 大小 | 秒/图 |
  |---|---|---|
  | 300 × 200 | 39.17 | 0.25 |
  | 1000 × 200 | 40.53 | 0.65 |
  | 3000 × 200 | 41.17 | 1.7 |
  | 10000 × 1000 | 41.84 | 11.4 |
  | 30000 × 1000 | 41.97 | 36 |

  KaMIS 是 43.15（CAM / RLD 论文）；每张 RB 图是若干个团拼成的，团的个数（test 平均 46.19）是 MIS 的上界。
- 用户的想法：DIFUSCO 的起点固定是 Ber(1/2)，改前向（加热前向，`20261117_difusco_rbsmall_heatflip`）对推理影响不大。把起点换成能量定义的分布（RLSA 采到的局部解），再用有监督的 flow matching / bridge matching 学“局部解 → KaMIS 解”，期望 MCMC 加学习好于两者各自。
- Shengyu 已试过的起点只有逐节点独立的 Bernoulli(p)（`/usr0/home/shengyuf/CAM4CO/experiments/fm_seeds_metrics.md`，RB-small）：p = 0.0887（等于标签密度）的 gap 2.79%，p = 0.5 的 gap 3.82%。他的薛定谔桥验证（`experiments/sb_rb_small/`）改的是前向路径，起点仍是 Ber(1/2)。

## 方法（代码：`cam/utils/bridge_paths.py`，`cam/solvers/fm.py`，UPSTREAM.md 第 43 条）

记 x₀ 为起点样本（RLSA 末状态），x₁ 为 KaMIS 标签，节点取值 0/1，t ∈ [0, 1]。

1. **Flow matching（λ = 0，混合路径，Gat et al. 2024）**
   - 路径：每个节点以概率 t 取 x₁ 的值，否则取 x₀ 的值。闭式，不用模拟。
   - 训练：网络读 (G, x_t, t)，输出 p_θ(x₁ = 1 | x_t, t)，对 x₁ 的交叉熵（同 DIFUSCO / 原 FM）。
   - 推理：从 [t₁, t₂]，节点翻转概率 = (t₂ − t₁)/(1 − t₁) · P(x₁ ≠ 当前值)，就是原 `fm.py` 的 Euler 步。
   - 速度公式里没有 x₀，所以对任意起点分布都成立（起点不需要无记忆）。
2. **Bridge matching（λ > 0，均匀翻转的参考过程）**
   - 参考过程：每个节点以速率 λ 翻转，P_s(翻) = (1 − e^{−2λs})/2。桥是把两端固定为 (x₀, x₁) 后的参考过程，单节点闭式：P(x_t = c | a, b) ∝ P_t(a → c) P_{1−t}(c → b)。
   - 推理（Doob h 变换）：rate = λ[q(1+e)/(1−e) + (1−q)(1−e)/(1+e)]，e = e^{−2λ(1−t)}，q = P(x₁ ≠ 当前值)。参考过程按节点分解，所以只用网络给的单节点边缘概率就精确。
   - λ → 0 退化成第 1 种；λ > 0 时路径上多出随机翻转（相当于熵正则）。这次用 λ = 0.5。
3. 每张图只有一个标签，独立配对就是唯一的配对，所以这一版还用不上薛定谔桥（求最优配对）。多个等大小的标签时才需要。

## 配置

- 数据：`/usr0/home/shengyuf/yunyi/data/rb_large_kamis/`。图是到 `/usr1/shengyuf/MIS/RB_large_*` 的软链接；标签由本地 KaMIS（redumis，每张图 60 秒，和 DIFUSCO 打 ER 标签的设定一致）在 11-17 夜里打的，在 `anno/` 下。
- 起点库（`cam/scripts/mis/build_start_bank.py`）：RLSA 1000 步，τ0 = 0.01，d = 5，取每条链的末状态。训练图每张 16 个；验证（前 64 张）和测试（1000 张）每张 20 个，等于推理链数，每条链从自己的起点出发。
- 训练：Shengyu 在 RB-large 上 FM 的配置。8 层 × 64，每步 40 张图（2 卡 × 20），AdamW，学习率 2e-4，200 个 epoch，不早停，每个 epoch 存权重，activation checkpoint，图缓存（`CAM_GRAPH_CACHE`）。
- 推理：50 步（cosine 时间表）× 20 条链，贪心解码，取最好。
- 流水线：`pipeline.sh`，依次等标签、建图缓存、建起点库、两组各 2 卡并行训练，再测 test。

## 测速（11-17，480 张图，FM，每步 40 张图）

| | 每步 | 验证 64 张 | 200 个 epoch |
|---|---|---|---|
| 不开图缓存，4 卡 | 8 秒 | — | 约 45 小时 |
| 开图缓存，1 卡 | 0.35 秒 | 约 55 秒 | 约 5 小时 |
| 开图缓存，4 卡 | 0.16 秒 | 约 15 秒 | 约 1.7 小时 |

networkx 解析一张 RB-large 图要约 3 秒，不开缓存时 GPU 空等。

## 评测要带的对照（之后补，不在这次流水线里）

1. 只跑 RLSA，总时间等于“起点 + 网络推理”：网络必须比把时间都给 MCMC 更好。
2. 起点库本身的最好解（起点库生成时已记录 best-of-K）。
3. 均匀起点的 FM（原 `fm.py`），以及 Shengyu 的 Bernoulli(p = 标签密度) 起点。
4. KaMIS（本地标签）。

## 状态

| run | 方法 | GPU | wandb |
|---|---|---|---|
| `20261118_fm_rblarge_mcmcstart_lam0_s0` | flow matching | 流水线自动分配 | — |
| `20261118_bm_rblarge_mcmcstart_lam0p5_s0` | bridge matching，λ = 0.5 | 流水线自动分配 | — |
