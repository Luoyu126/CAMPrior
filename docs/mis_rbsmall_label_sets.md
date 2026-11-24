# RB-small MIS：构造过的多解标签集

写给接手的人或 agent。说明 RB-small 最大独立集（MIS）上，每张训练图构造过哪些“多个不同解”的标签集：
- 来源：KaMIS、RLSA、DiffUCO、CAM、DIFUSCO 学生；
- 每套怎么构造、文件在哪、统计量；
- 用在哪些实验、结果如何；
- 已知的坑。

统计量都由 `runs/scratch/labelsets_1124/stats.py` 在 11-24 重新算过。

**先看第 5 节**：11-24 发现多种子 KaMIS 标签集的节点顺序错了，用到它的结论要作废重做。

## 1. 背景

- 图：`/usr1/shengyuf/MIS/RB_small_{train,valid,test}`（RB-[200-300]）。训练图 4000 张，验证 500 张（训练中只用前 64 张挑 checkpoint），测试 1000 张。
- 单个 KaMIS 标签：每张图一个，在 `anno/<stem>_unweighted.result`，平均大小 20.162，是 DIFUSCO 原文的标签。
- 一张 RB 图通常有几十到几百个同样大的最大独立集。KaMIS 给哪一个，取决于它自己的随机性。

### 文件格式

所有多解文件都是 `torch.save` 的字典：
- `files`：图文件名，按 `sorted(glob)` 排序，与加载器一致；
- `sets`：每张图一个 `uint8` 张量，形状为（解的个数 × 节点数）；
- `meta`：构造参数。

`ul_samples.py` 的原始采样文件是另一种格式：`{'files', 'samples': [{'sizes', 'sets', 'counts', 'kamis'}]}`，保存每张图全部不同的样本及其出现次数，不只是最大的。

训练时用 `--label_set FILE --label_choice {first|random|soft}`：
- `first`：用第一个解；
- `random`：每步随机抽一个；
- `soft`：后验均值软目标（UPSTREAM 第 56 条，见 `docs/20261116/difusco_sota_soft_labels.md` §2.3）。

读取在 `cam/utils/label_sets.py`。

### 节点顺序（重要）

`cam/datasets/mis.py` 按 gpickle 里 **G.nodes() 的插入顺序**编号节点，而 `anno/*.result` 和 redumis 按**排序后的节点 id** 编号；读 anno 标签时加载器会重新对齐（上游 1d3f4ae）。

所以：
- 经过加载器（DiffUCO / CAM / RLSA / DIFUSCO 的采样）得到的解，天然是加载器的顺序，没有问题；
- 直接调用 redumis 得到的解，必须先换成加载器的顺序。

## 2. 标签集一览（4000 张训练图）

表中各列的含义：
- **个数**：每图平均个数（中位数）。
- **大小**：解的平均大小。
- **Hamming**：同一张图内两两之间的平均 Hamming 距离。只算前 500 张图，每图最多取 300 个解；一个解约 20 个点，距离 40 就是完全不重叠。
- **与 KaMIS 共享**：每个解平均有多大比例的点出现在 KaMIS 标签里（共享点数 ÷ KaMIS 标签大小）。
- **含 KaMIS 标签**：解集里包含 KaMIS 标签本身的图所占比例。

| 来源 | 文件（`/usr0/home/shengyuf/yunyi/data/` 下） | 个数 | 大小 | Hamming | 与 KaMIS 共享 | 含 KaMIS 标签 |
|---|---|---|---|---|---|---|
| KaMIS 单标签 | `anno/*.result`（图目录内） | 1 | 20.162 | 0 | 1 | 100% |
| RLSA 最优解 | `rb_small_solsets/train.pt` | 178.7（128） | 20.162 | 22.8 | 0.366 | 100%（标签放第一个） |
| KaMIS 多种子 | `rb_small_kamis_multi/k128_loaderorder.pt`（**原 `k128.pt` 顺序错**） | 66.9（76） | 20.162 | 19.4 | 0.511 | 100% |
| DiffUCO，200 个样本的全部最大解 | `rb_small_ullabels/train_diffuco_all200_top.pt` | 57.0（32） | 20.082 | 21.5 | 0.323 | 0.6% |
| DiffUCO，1000 个样本的全部最大解 | `rb_small_div/d1000_all.pt` | 243.2（106） | 20.124 | 21.2 | 0.340 | 1.8% |
| DiffUCO，3000 个样本的全部最大解 | `rb_small_div/d3000_all.pt` | 648.3（224） | 20.142 | 20.9 | 0.350 | 2.9% |
| CAM，1000 个样本的全部最大解 | `rb_small_div/c1000_all.pt` | 171.0（52） | 20.122 | 20.0 | 0.359 | 3.0% |
| RLSA ∪ DiffUCO-1000 | `rb_small_strong/s1000_any_fps1000.pt` | 378.8（244） | 20.162 | 22.7 | 0.366 | 99.6% |
| 个数对齐：KaMIS | `rb_small_kamis_multi/matched_kamis_loaderorder.pt`（**原文件顺序错**） | 42.1（29） | 20.162 | 18.9 | 0.512 | 71% |
| 个数对齐：DiffUCO | `rb_small_kamis_multi/matched_diffuco.pt` | 42.1（29） | 20.124 | 19.3 | 0.341 | 1.8% |
| 个数对齐：CAM | `rb_small_kamis_multi/matched_cam.pt` | 30.7（18） | 20.122 | 17.6 | 0.358 | 1.9% |
| 个数对齐：DiffUCO（按 CAM 的个数） | `rb_small_kamis_multi/matched_diffucoc.pt` | 30.7（18） | 20.124 | 18.9 | 0.340 | 1.5% |
| 参照：DiffUCO 单标签，20 个样本取最大 | `rb_small_ullabels/train_diffuco.pt` | 1 | 19.907 | — | 0.279 | 0.1% |
| 参照：DiffUCO 单标签，200 个样本取最大 | `rb_small_ullabels/train_diffuco_k200.pt` | 1 | 20.079 | — | 0.320 | 0.3% |

几点观察：
- **KaMIS 的解彼此更像，也更像 KaMIS 标签。** 不同种子的 KaMIS 解两两 Hamming 19.4，与 KaMIS 标签共享 51%；DiffUCO 的解共享约 34%，几乎从不包含 KaMIS 标签。两类求解器落在最优解空间的不同区域。
- **RLSA 找到的最优解最分散**：Hamming 22.8，与 KaMIS 标签共享 37%。
- **DiffUCO / CAM 的最大解比 KaMIS 小约 0.04。** 约 2–4% 的图上，1000 个样本都没采到 KaMIS 的大小。

## 3. 各来源怎么构造

### 3.1 RLSA 最优解集合（11-18）

- 脚本：`cam/scripts/mis/build_solution_sets.py`（UPSTREAM 第 44 条）。
- 构造：每图 500 条 RLSA 链，每条 3000 步，τ0 = 0.01 线性降到 0，每步取前 d = 5 个翻转，随机初始化，种子 0。每条链的终态贪心解码，保留与最好大小相同的不同解；KaMIS 标签放第一个。
- 规模：10 张卡，约 6.5 分钟，分片在 `train_part*.pt`。
- 结果：
  - 没有一张图 RLSA 超过 KaMIS；
  - 184 张图（4.6%）RLSA 没到 KaMIS 的大小，只剩 KaMIS 标签一个。
- 解的个数是**很松的下界**：DiffUCO 的 20 样本标签在约 3/4 的图上达到最优大小，但只有 38 张图的这个解出现在 RLSA 集合里。
- 派生文件：`train_rule.pt`，每图在最优解里按“所选点度数和最小，再比邻居度数和”挑一个（`cam/scripts/mis/rule_labels.py`）。

### 3.2 KaMIS 多种子（11-20）

- 脚本：`cam/scripts/mis/kamis_multi.py`（UPSTREAM 第 58、66 条）。
- 构造：
  - 每图跑 redumis 128 次，种子 0–127，每次限时 0.2 秒；
  - 连同 anno 里原有的 KaMIS 标签，保留最好大小的所有不同解。
- 结果：几乎每个种子给出一个不同的最大解，每图平均 66.9 个，大小全部等于 KaMIS 标签。
- **坑**：11-24 之前生成的 `k128.pt` 用的是 redumis 的排序节点顺序（见第 5 节）。已修脚本；`*_loaderorder.pt` 是把旧文件按列重排的副本，同一批解，已验证全部是合法独立集，且包含 KaMIS 标签。

### 3.3 DiffUCO 采样（11-18 起）

- 模型：DiffUCO RB-small 检查点 `20260513_diffuco_rbsmall_annealing` 的 best.pt，测试 19.88。
- 采样协议：用它自己的测试协议在训练图上采样，即 50 步翻转链，每个样本用贪心解码成独立集。
- 单标签：`cam/scripts/mis/ul_labels.py`（UPSTREAM 第 46 条），每图取 K 个样本里最大的一个。
  - `train_diffuco.pt`：K = 20，平均 19.907；
  - `train_diffuco_k200.pt`：K = 200，种子 1，平均 20.079；
  - `train_diffuco_k200_fill.pt`：在 k200 基础上，把没到 KaMIS 大小的 330 张图换成 RLSA 集合里离它最近的最优解（`polish_labels.py --k 0 --pool`），平均 20.162。
- 全部样本：`cam/scripts/mis/ul_samples.py`（UPSTREAM 第 47 条），保存每图全部不同样本及其次数。
  - `train_diffuco_all200.pt`：种子 2；
  - `train_diffuco_all1000.pt`：种子 7；
  - `_s11.pt`、`_s12.pt`：另外两次 1000 个；
  - `train_diffuco_all3000.pt`：上面三份合并。
- 取最大并挑选：`cam/scripts/mis/diverse_labels.py`（UPSTREAM 第 53 条），取每图大小最大的全部不同解（`*_all`），或在其中按 Hamming 最远点采样挑 M 个（`*_fpsM`，从中心点开始）。
  - `rb_small_div/d200_*`、`d1000_*`、`d3000_all`；
  - `rb_small_ullabels/train_diffuco_all200_top.pt` 等价于 d200_all。

### 3.4 CAM 采样（11-21）

- 模型：Shengyu 的 `cam_d4` 模型，测试 19.907。
- 构造：每图 1000 个样本，种子 7（`ul_samples.py --method CAM`，两卡分片 `train_cam_all1000_s{0,1}.pt`，合并为 `train_cam_all1000.pt`）。再用 `diverse_labels.py` 得到 `rb_small_div/c1000_all.pt`。

### 3.5 个数对齐（11-20、11-21）

- 脚本：`cam/scripts/mis/match_label_counts.py`，种子 0。
- 构造：每张图上，每个来源都随机保留同样多的解，个数取各来源中最少的那个。这样比较来源时不受“每图标签个数”影响。
- 产物：
  - `matched_kamis` / `matched_diffuco`：KaMIS k128 与 DiffUCO d1000_all 对齐；
  - `matched_cam`：CAM 截到 `matched_diffuco` 的个数，CAM 不够的图就少取；
  - `matched_diffucoc`：DiffUCO 再截到 CAM 的个数，作对照。

### 3.6 由上面派生的挑选方式（都在 `experiments/20261119_difusco_rbsmall_reach/`）

| 名字 | 文件 | 规则 | 脚本 |
|---|---|---|---|
| dselM{1,4,16} / dselbot4 | `rb_small_probe/dsel_*.pt` | DiffUCO 200 个样本的最大解里，一步探针预测得最好的 M 个 / 最差的 4 个 | `probe_select.py` |
| t1000_tau{0.02,0.05} | `rb_small_typ/` | 好学阈值：DiffUCO 采到概率 ≥ τ 的最大大小；最大解太罕见的图退到小一号的解 | `typical_labels.py` |
| s1000_{any,learnK}_fpsM | `rb_small_strong/` | RLSA ∪ DiffUCO-1000 的池子，按到最近 DiffUCO 样本的距离挑最好学的 K 个，再最远点 M 个 | `strong_labels.py` |
| c1_{far,near}16_fps4 | `rb_small_cov/` | DiffUCO 最大解里，离 k200 学生样本最远 / 最近的 16 个，再最远点 4 个 | `coverage_labels.py` |
| r1_{self,top,union} | `rb_small_reach/` | k200 学生自己 50 步 200 个样本的最大解（中心点 / 全部 / 与 k200 取大） | `reach_labels.py` |

## 4. 用在哪些实验、结果

测试协议：1000 张测试图，50 步，每图 20 个样本取最大，renoise。DiffUCO 在这个协议下为 19.880。
数字是“训练种子 × 测试种子”的平均。细节见各 README。

| 标签 | 训练 | renoise | 出处 |
|---|---|---|---|
| KaMIS 单标签 | 1000 步扩散，200 epoch | 19.669 | `experiments/20261118_difusco_rbsmall_multilabel` |
| RLSA 最优解，度数规则挑一个 / 模型自己挑最近的 | 同上 | 19.577 / 19.633 | 同上 |
| DiffUCO 单标签（20 / 200 / 补齐） | 同上 | 19.712 / 19.739 / 19.750 | `experiments/20261118_difusco_rbsmall_ullabel` |
| DiffUCO 200 个样本取最大（k200） | 50 步扩散，200 epoch | 19.828 | `experiments/20261119_difusco_rbsmall_fewstep` |
| DiffUCO 全部最大解 all200_top，随机抽 | 同上 | 19.842 | `experiments/20261119_difusco_rbsmall_reach` §4 |
| d1000_all，随机抽 / 软目标 | 400 epoch | 19.866 / 19.875 | 同上 §6 |
| **d1000_all，软目标** | **800 epoch** | **19.890** | 同上 §7；`docs/20261116/difusco_sota_soft_labels.md` |
| d3000_all，软目标 | 800 epoch，2 个种子 | 19.914 | 同上 §7 补充 |
| RLSA ∪ DiffUCO（上限 = KaMIS），软目标 | 400 epoch | 19.84–19.86 | 同上 §6 r6 |
| matched_diffuco，随机抽 / 软目标 | 400 epoch | 19.871 / 19.884 | 同上 §8 |
| matched_cam，随机抽 / 软目标 | 400 epoch | 19.853 / 19.872 | 同上 §9 |
| ~~matched_kamis，随机抽 / 软目标~~ | 400 epoch | ~~19.536 / 19.581~~ **作废**（标签顺序错） | 同上 §8 |

与多解有关的主要结论（不涉及 matched_kamis 的部分仍然成立）：
- 标签风格比大小重要：DiffUCO / CAM 的解比 KaMIS 小，但更好学。证据来自单标签对比（19.712 起 vs 19.669）和一步探针（交叉熵 0.261 vs 0.273）。
- 每图标签越多、越多样越好；只挑最好预测的少数几个反而更差。
- 软目标让“很多标签”可用：种子间标准差从 0.019 降到 0.003。
- 加入风格不同的 RLSA 最优解，虽然把上限抬到 KaMIS 的水平，但并不更好。
- 按最优解个数分组时，SL 落后最多的是最优解少的图，所以“多个最优解被交叉熵平均”不是主因（multilabel README 附录）。

## 5. 已知问题：多种子 KaMIS 的节点顺序（11-24 发现）

### 问题

`kamis_multi.py` 直接保存 redumis 的输出（排序节点顺序），而加载器用插入顺序。按加载器的顺序检查（`runs/scratch/labelsets_1124/check_order.py`，前 100 张图）：
- `k128.pt` 和 `matched_kamis.pt` 中 98.3% 的解不是独立集；
- 其余所有 MIS 标签集都是 0%。

所以 11-20 的 A / B 两组（`q5_mkamis_long[_soft]`）实际是在打乱的标签上训练的。

### 作废的结论

- A / B 的 19.536 / 19.581，以及据此得到的差值：
  - C − A = +0.336、D − B = +0.303；
  - CAM − KaMIS = +0.318 / +0.291；
  - 软目标在 KaMIS 上 +0.045。
- `docs/20261116/sl_vs_ul_findings_1121.md` §1.1、§1.2 里 A / B 两行，包括 renoise_decode 在 KaMIS 上 +0.2 的结论。
- 同文中“KaMIS 多解反而比单个 KaMIS 标签更难学”这个现象，很可能就是这个错误造成的。

### 不受影响的结论

- 单 KaMIS 标签的所有实验：加载器自己读 anno，并且会重新对齐。
- 所有 DiffUCO、CAM、RLSA 标签集。
- MDS 的 ILP 标签：已检查 `data/mds_labels/*`，全部是合法支配集。

### 修复与待办

- 已修：commit `7173880`，`kamis_multi.py` 在保存前把列换成加载器的顺序，并写入 UPSTREAM 第 66 条。
- 重排后的副本：`k128_loaderorder.pt`、`matched_kamis_loaderorder.pt`，同一批解，已验证。
- 待办：用 `matched_kamis_loaderorder.pt` 重跑 A / B 两组（400 epoch，2 个种子，各约 4 小时/卡），才能重新回答“同样个数下，KaMIS 多解与 DiffUCO 多解谁更好学”。需要用户批准后再启动。`launch.sh` 里 `mkamis` 模式的路径要改成新文件。

## 6. 相关脚本速查

| 脚本 | 作用 |
|---|---|
| `cam/scripts/mis/build_solution_sets.py` | RLSA 最优解集合 |
| `cam/scripts/mis/kamis_multi.py` | KaMIS 多种子 |
| `cam/scripts/mis/ul_labels.py` | DiffUCO K 个样本取最大（单标签） |
| `cam/scripts/mis/ul_samples.py` | DiffUCO / CAM 全部样本 |
| `cam/scripts/mis/diverse_labels.py` | 取最大的全部解 / 最远点采样 |
| `cam/scripts/mis/match_label_counts.py` | 各来源每图个数对齐 |
| `cam/scripts/mis/polish_labels.py` | 单标签补齐到 KaMIS 大小 |
| `cam/scripts/mis/{probe_select,typical_labels,strong_labels,coverage_labels,reach_labels}.py` | 第 3.6 节的派生挑选 |
| `cam/utils/label_sets.py` | 训练时读取；`posterior_mean` 是软目标 |
| `runs/scratch/labelsets_1124/{stats,check_order,reorder_kamis}.py` | 本文的统计、顺序检查、重排（未提交） |
