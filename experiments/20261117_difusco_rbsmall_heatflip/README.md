# 20261117_difusco_rbsmall_heatflip：加热前向 DIFUSCO，不早停的完整训练（RB-[200-300]）

## 目的

11-16 那组（`20261116_difusco_rbsmall_heatflip`）用 patience 20 早停。λ = 0 在第 49 个 epoch 取到最好、第 69 个 epoch 就停了，当时学习率还有峰值的约 86%。验证只有 64 张图，单个 epoch 的波动约 ±0.2，早停时机受噪声影响很大。这次三组（本文件夹的 λ = 0.1、λ = 0，以及 `20261117_difusco_rbsmall_randflip`）都不早停，跑满 200 个 epoch，并且每个 epoch 存一份权重（用户 11-17 定）。

## 配置

- 和 `20261116_difusco_rbsmall_heatflip` 相同：加热前向（λ_t 在 t = 500 处退到 0），验证和测试用重新加噪采样器，AdamW，学习率 2e-4 按 cosine 衰减，200 个 epoch，推理 50 步 × 20 条链，验证 64 张，种子 0。
- 不同：
  - `--patience 1000`：大于 200，所以不会早停；
  - `--save_every_epoch`：每个 epoch 存 `epoch_XXXX.pt`；best.pt 仍按验证集保存；
  - 4 张卡数据并行，每卡 10 张图，每步合计 40 张，和 DiffUCO 一致；验证每卡 16 张，合计正好 64 张。
- `NCCL_IB_DISABLE=1`：sa 上 NCCL 探测 InfiniBand 时会段错误，11-17 用两进程 all-reduce 测试确认，设了这个变量就正常。
- 和单卡 batch 40 的区别：loss 先在每张卡内对节点求平均，再在卡之间平均，节点的权重略有不同。
- 测速（11-17，λ = 0.1，负载约 40）：单卡每个 epoch 约 7.2 分钟，4 卡约 2.7 分钟。验证按卡数线性加速；训练 4 卡约快 1.9 倍。

## 状态

| run | GPU | 开始（UTC） | wandb |
|---|---|---|---|
| `20261117_difusco_rbsmall_heatflip_lam0p1_s0` | 1–4 | 2026-11-17 15:51 | [4ossajux](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/4ossajux) |
| `20261117_difusco_rbsmall_heatflip_lam0_s0` | 5–8 | 2026-11-17 15:51 | [bl9qbv4m](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/bl9qbv4m) |

11-16 那组的 λ = 0.1（GPU 0，patience 20）照常跑，不停。
