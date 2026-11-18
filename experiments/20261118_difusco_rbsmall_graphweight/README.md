# 20261118_difusco_rbsmall_graphweight：按图加权 DIFUSCO 的训练损失（RB-small）

## 动机

`20261118_difusco_rbsmall_multilabel` 的附录里，4000 张训练图上 SL（DIFUSCO）落后 UL（DiffUCO）最多的是最优解少的图：只有一个最优解的 184 张图上差 0.73（renoise 0.36），最优解多于 250 个的图上两者都接近最优。SL 在这些难图上的交叉熵也最高（一个最优解 0.185，多于 450 个 0.160），而它们在训练集里只占一小部分（≤ 30 个最优解的约 23%）。这里看把训练比重挪到难图上能不能缩小差距。

## 文献

- DIFUSCO（Sun & Yang 2023，附录训练细节）：只按问题规模做课程（TSP-500 从 TSP-100 的权重开始），不按实例难度。
- Annealed Training（Sun et al. 2022，arXiv 2207.11542）/ DiffUCO：温度退火看作从易到难的课程，调的是温度，不是实例权重。
- Hardness-Adaptive Curriculum（Zhang et al., AAAI 2022，arXiv 2204.03236，`ref/lit_pdfs/2204.03236.txt`）：TSP-50、REINFORCE；难度 = 当前模型再做几步梯度下降还能改进多少（式 7），批内权重 w = softmax(F(H)/T)（式 12），损失 Σ w_i L_i（式 13）。主要收益来自生成更难的实例，加权本身只被描述为“降低方差、收敛更快”（图 4），没有单独的数值。

## 设置

除损失权重外和 KaMIS 标签的 baseline（`20261117_difusco_rbsmall_heatflip` λ=0，及 `20261118_difusco_rbsmall_multilabel` 的 kamis s1）完全一样：8×64，1000 步均匀前向，每步 40 张图（2 张卡各 20），AdamW 2e-4 余弦，200 个 epoch，不早停，验证 64 张图用 renoise。损失 = Σ_i w_g(i) ce_i / Σ_i w_g(i)（cam/UPSTREAM.md 第 45 条）。

| 组 | 权重 | 有效样本量 |
|---|---|---|
| nsets | w = M^(-1/2)，M 为该图的最优解个数（500 条 RLSA 链找到的，`data/rb_small_solsets/train.pt`），固定 | 约 38%（1533 / 4000）；只有一个最优解的图占比 4.6% → 26% |
| ce | w = exp(z)，z 为标准化后的 log 难度，截断到 ±3；难度 = 该图交叉熵 / 同一噪声段（t 分 20 段）的滑动平均交叉熵，按该图历次访问做 EMA（0.5）；从上一次访问得到，相当于晚一个 epoch | 冒烟测试中 23–43% |

种子 0、1；nsets 用 GPU 0,1 / 2,3，ce 用 4,5 / 6,7。训练完后各自在 1000 张测试图上测 D3PM 后验和 renoise × 测试种子 0、1（`test_after_train.sh`）。

启动：2026-11-18 09:48（匹兹堡时间）。每个 epoch 约 21 秒。
