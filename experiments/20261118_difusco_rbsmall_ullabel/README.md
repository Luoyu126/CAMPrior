# 20261118_difusco_rbsmall_ullabel：用 DiffUCO 的解当标签训练 DIFUSCO（RB-small）

## 动机

前面的实验排除了标签选法不一致（H2）、训练图数量（H3）、训练比重（`20261118_difusco_rbsmall_graphweight`）。这里检验“标签好不好学”：机器翻译里非自回归模型同样每个位置独立输出、面对多种正确译法，最有效的标准做法是用强教师模型的输出当训练目标（序列级知识蒸馏；Zhou et al. 2020，arXiv 1911.02727），教师的输出对网络来说更“一致”。

- 如果 SL 在 UL 的输出上训练就能接近 UL，说明差距在于 KaMIS 标签对网络不自然、难学。
- 如果仍然学不到，差距在训练目标或训练时看到的状态。

## 标签

`cam/scripts/mis/ul_labels.py`（cam/UPSTREAM.md 第 46 条）：用 DiffUCO 的 RB-small 模型（`20260513_diffuco_rbsmall_annealing` 的 best.pt，测试 19.88）按它自己的测试协议（50 步 × 20 个样本）在 4000 张训练图上采样，每张图取 20 个样本里贪心解码后最大的一个。输出 `data/rb_small_ullabels/train_diffuco.pt`。

- 平均大小 19.907（KaMIS 标签 20.162）；75.1% 的图上不小于 KaMIS 标签；全部是合法独立集。
- 和 KaMIS 标签平均只共享 27.9% 的点；只有 2 张图和 KaMIS 标签完全相同。
- 只有 38 张图的 UL 标签出现在 `data/rb_small_solsets` 的最优解集合里，而 UL 标签在约 3/4 的图上达到最优大小：那份集合（500 条 RLSA 链）漏掉了大部分最优解，最优解个数是很松的下界。

## 设置

除标签外和 KaMIS 标签基线完全一样（8×64，1000 步均匀前向，每步 40 张图，AdamW 2e-4 余弦，200 个 epoch，不早停，验证用 renoise），`--label_set .../train_diffuco.pt --label_choice first`。种子 0（GPU 0,1）、1（GPU 2,3），训练完各测 D3PM 后验 / renoise × 测试种子 0、1。

启动：2026-11-18 12:21（匹兹堡时间）。
