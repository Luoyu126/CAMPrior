# 交接：DIFUSCO 的 renoise 采样器

写给接手的 agent。说明 renoise 是什么、代码在哪、怎么用、已知结论和坑。行号以 commit `be0a255` 为准，改代码后请重新 grep。

## 1. 一句话

DIFUSCO 每个去噪步先让网络从 x_t 预测 x̂₀ 的概率，然后：

- **posterior**（原版 DIFUSCO，默认）：用 D3PM 后验 q(x_{t−1} | x_t, x̂₀) 走到下一步，保留当前 x_t 的信息。
- **renoise**：从预测里抽一个 x̂₀ ~ Bernoulli(p)，再用**和训练完全相同的前向加噪**把它直接加噪到下一个时间步 t₂，不看当前 x_t。

renoise 就是 Fast T2T 的 Alg. 1，等价于 DDIM σ=0 的离散版。它**不是我们的方法**，用户已定：作为主采样器使用，引用 Fast T2T，同时报告 posterior 结果。见用户记忆 "Sampler choice: renoise main, cite Fast T2T"。

**训练不受采样器影响**：
- 训练（`train_batch`）只用前向 q(x_t | x₀) 加噪、预测 x₀ 的交叉熵。D3PM 后验只出现在采样里。
- `--difusco_sampler` 只在两处起作用：
  - 训练中每轮的验证（它决定 `best.pt` 选哪一轮）；
  - 测试。

## 2. 代码位置（都在 `cam/` 下）

| 内容 | 位置 |
|---|---|
| 命令行开关 `--difusco_sampler {posterior, renoise, renoise_decode, renoise_feasible}` | `main.py:69` |
| TSP 解码后 2-opt 次数 `--decode_two_opt N` | `main.py:71` |
| 无监督（CAM/DiffUCO）翻转链里的对应物 `--ul_chain_decode` | `main.py:70`，`solvers/meta_solver.py:410`（`chain_decode_tour`）、`:450` |
| 读取开关：`self.sampler`、`self.decode_two_opt`、`self.heated` | `solvers/difusco.py:22-25` |
| D3PM 后验采样 `categorical_posterior` | `solvers/difusco.py:82-119` |
| 训练 `train_batch`：前向加噪 + x₀ 交叉熵，与采样器无关 | `solvers/difusco.py:162-197` |
| **renoise 本体 `renoise(graph, x0_pred_prob, t2)`** | `solvers/difusco.py:239-259` |
| 采样主循环 `evaluate_single`：按 `self.sampler` 分支 | `solvers/difusco.py:304-336`（分支在 `:328-331`） |
| 节点问题的"解码后再加噪" `decode_x0` | `solvers/difusco.py:294-302` |
| TSP 的"解码后再加噪" `decode_tour`（贪心拼回路 + 2-opt） | `solvers/difusco.py:281-292` |
| 诊断 `trace_inputs`（环境变量 `CAM_TRACE_INPUTS=FILE`） | `solvers/difusco.py:261-279` |
| 节点问题的前向加噪 `HeatedForward.sample`（`heat_lambda=0` 时就是 DIFUSCO 的均匀前向，闭式一跳） | `utils/heated_forward.py:60-75` |
| 推理时间表 `DiffusionInferenceSchedule.__call__`（给出每步的 t1→t2） | `utils/diffusion_schedulers.py:85-` |
| 改动记录 | `UPSTREAM.md` 第 42 条（renoise）、第 59 条（renoise_decode / feasible、trace）、第 60 条（边变量）、第 62 条（TSP decode）、第 63 条（ul_chain_decode） |

### renoise 的逻辑（`solvers/difusco.py:239-259`）

```
p1 = x̂₀ 为 1 的概率
if t2 == 0: return p1          # 最后一步返回概率，交给 problem.evaluate 做贪心解码（与 posterior 相同）
x0 = bernoulli(p1)
边变量（TSP）:
    if renoise_decode: x0 = decode_tour(...)        # 拼成回路 + decode_two_opt 次 2-opt
    return diffusion.sample(one_hot(x0), t2)         # 训练用的同一个前向
节点变量（MIS / MDS / MaxCut）:
    if renoise_decode / renoise_feasible: x0 = decode_x0(...)
    return heated.sample(A, b, x0, t2)               # heat_lambda=0 时 = 训练前向的闭式边际
```

关键不变量：renoise 给出的 x_{t₂} 的分布，和训练时"标签加噪到 t₂"的形式一致，前提是 x̂₀ 是合法解。所以 `heat_lambda` 等前向参数必须和训练时一致，测试脚本里要写同样的 `--heat_lambda 0 --heat_until 0.5`。

### 三个变体

- `renoise`：上面的普通版。
- `renoise_decode`：
  - 加噪前先把 x̂₀ 变成合法解：节点问题用问题自带的贪心解码（抽中的节点优先），得到极大 / 极小合法解；TSP 拼回路，再做 `--decode_two_opt` 次 2-opt。
  - 每步多一次 Python 循环的贪心，耗时约为 renoise 的 2 倍。
  - TSP 上它就是 IDEQ / PCI 的链内 2-opt，也不是我们的方法。
- `renoise_feasible`：只去掉冲突节点、不补点。仅节点问题；在 MIS 上更差，基本不用。

## 3. 怎么用

训练和测试都加一个开关即可。例子见 `experiments/20261123_difusco_mds_labels/{launch.sh,test.sh}`：

```bash
# 测试一个已训练的 DIFUSCO 节点问题模型（MDS 为例），三种采样器各测一次
bash experiments/20261123_difusco_mds_labels/test.sh <gpu> <run 名> posterior
bash experiments/20261123_difusco_mds_labels/test.sh <gpu> <run 名> renoise
bash experiments/20261123_difusco_mds_labels/test.sh <gpu> <run 名> renoise_decode
# 本质是 main.py 测试参数里加： --difusco_sampler renoise   （其余参数与训练一致：--num_tp --diffusion_schedule
#   --heat_lambda --heat_until，以及推理的 --num_t --num_k --inference_schedule）
```

TSP：

```bash
--difusco_sampler renoise_decode --decode_two_opt 20
```

见 `experiments/20261122_difusco_tsp500_chain2opt/test.sh`。

训练时 `--difusco_sampler` 只影响验证和 `best.pt` 的选择。比较不同采样器时，要在同一个 checkpoint 上换采样器测，不要拿"用不同验证采样器选出的 best.pt"互相比。

诊断"采样输入是否像训练输入"：

```bash
CAM_TRACE_INPUTS=out.jsonl <测试命令 --test_samples 100 ...>
```

每张图写一行，每步给出采样器输入和训练形式输入（本链最终解按训练前向加噪到同一 t）的选中数、相邻同为 1 的边数、预测对最终解的交叉熵。注意第二项是 MIS 专用的统计。

## 4. 已有结论（都是 1000 张测试图，逐图配对，95% 区间；细节见各 README）

| 问题 / 数据 | renoise − posterior | renoise_decode − renoise | 出处 |
|---|---|---|---|
| MIS RB-small，DiffUCO 标签（好学） | renoise 显著更好 | ≈ 0 | `docs/20261116/sl_vs_ul_findings_1121.md` §1 |
| MIS RB-small，KaMIS 标签（难学） | renoise 更好 | 约 +0.2（MIS 越大越好，即更好） | 同上 §1.2 |
| MDS BA-small（标签集中、好学） | −0.15 到 −0.21（更好） | +0.02 到 +0.06（略差） | `experiments/20261123_difusco_mds_labels/README.md` |
| MDS RB-small（标签分散、难学） | −0.02 到 +0.06（不再更好） | −0.17 到 −0.25（大幅更好，gap 约 5% → 约 2.7%） | 同上 |
| TSP-500 | 更好 | 链内 2-opt：d≈20 饱和，步数 T 是主轴（T200 0.485% vs 基线 1.415%） | `experiments/20261122_difusco_tsp500_chain2opt/README.md` |

目前的解读：
- **标签好学时**，普通 renoise 就够，优势来自每步重新决定、纠错。轨迹诊断里，两种采样器的一阶输入统计都和训练形式一致。
- **标签难学时**，网络抽出的 x̂₀ 常常不合法，先解码再加噪把输入拉回训练形式，收益大。这是"采样走训练见过的状态"目前最直接的证据。

## 5. 坑

1. **最后一步**：`t2 == 0` 时 renoise 返回概率，不是 0/1 样本。最终解由 `problem.evaluate` 贪心解码，与 posterior 路径相同，不要在外面再抽样。
2. **前向一致**：节点问题走 `HeatedForward`，边问题走 `CategoricalDiffusion.sample`。如果训练时改了 `--heat_lambda`、`--diffusion_schedule` 或 `--num_tp`，测试必须一致，否则 renoise 加噪的分布和训练不同。
3. **网络看不到 t**：MIS / MDS 用的 `EncodeProcessDecode` 忽略 time 参数（`models/enprocessde.py:114-126`），只能从 x_t 的密度推断噪声水平。这是沿用 CAM 论文 B.2 的设置（"We omit the timestep embedding in all models"，`docs/20261116/papers/2605.30920.txt:2290`）。TSP 的 AnGNN 是看得到 t 的。
4. **`renoise_decode` 依赖问题的 `evaluate`**：它会把 x̂₀ 解成合法的极大 / 极小解。MDS 的 `evaluate` 是先贪心覆盖再删冗余，得到极小支配集（`envs/mds.py`）。新问题要先确认它的 `evaluate` 返回 `(obj, solution)`，且 solution 的布局是 N×K。
5. **耗时**：
   - renoise 与 posterior 同速；
   - `renoise_decode` 约 2 倍（节点问题），TSP 随 `--decode_two_opt` 变；
   - 同等时间比较时，renoise 可以多采样本（MIS 上 40 个样本几乎不加时间）。
6. **不是我们的贡献**：renoise 引用 Fast T2T；链内解码引用 IDEQ（arXiv 2412.13858）和 PCI（arXiv 2606.09343）。原文在 `/usr0/home/shengyuf/yunyi/ref/lit_pdfs/`。
