# 20261117_difusco_rbsmall_randflip：均匀前向 DIFUSCO，不早停的完整训练（RB-[200-300]）

## 目的

和 `20261117_difusco_rbsmall_heatflip` 配套的 D3PM 一格：均匀翻转前向，D3PM 后验采样器，跑满 200 个 epoch，不早停（用户 11-17 定）。原因见那边的 README。

## 配置

- 和 `20261116_difusco_rbsmall_randflip` 相同：单卡、batch 40，AdamW，学习率 2e-4 按 cosine 衰减，200 个 epoch，推理 50 步 × 20 条链，验证 64 张，种子 0。
- 不同：`--patience 1000`，大于 200，所以不会早停；`--save_every_epoch`，每个 epoch 存一份权重。

## 状态

| run | GPU | 开始（UTC） | wandb |
|---|---|---|---|
| `20261117_difusco_rbsmall_randflip_s0` | 9 | 2026-11-17 15:51 | [3zdeizwh](https://forge.coreweave.com/wandb/yunyi-organization/SLVUL/runs/3zdeizwh) |

## 结果（test，1000 张图，推理种子 0 / 1）

训练还没结束时，验证曲线已平（第 100 个 epoch 以后每 20 个 epoch 的均值在 19.54–19.58），用户 11-17 定先测当时的 best.pt：第 104 个 epoch（验证 19.67）。权重快照和结果在 run 目录的 `best_ep0104/`。脚本 `experiments/20261117_difusco_rbsmall_heatflip/test.sh <gpu> randflip <sampler> <seed> <快照目录>`。

| 不早停，best.pt 来自 | D3PM 后验采样 | 重新加噪采样 |
|---|---|---|
| **randflip（第 180 个 epoch，跑满 200 个 epoch 后的 best.pt，验证 19.73）** | **19.333 / 19.362** | **19.653 / 19.677** |
| randflip 训练中途快照（第 104 个 epoch，按 D3PM 后验选权重） | 19.303 / 19.303 | 19.597 / 19.612 |
| λ = 0（第 179 个 epoch，按重新加噪选权重） | 19.358 / 19.349 | 19.658 / 19.647 |
| λ = 0.1（第 164 个 epoch，按重新加噪选权重） | 19.292 / 19.276 | 19.587 / 19.580 |

- 11-18 训练跑满后 best.pt 换成第 180 个 epoch（验证 19.73 > 19.67），自动测试（`experiments/20261117_difusco_rbsmall_heatflip/test_after_train.sh`）结果写在 run 目录。它和 λ = 0 几乎一样（差 0.01 以内），λ = 0.1 比两者都低约 0.07–0.08，四格方向一致。
- randflip 和 λ = 0 的训练完全相同（均匀前向），只差验证时用哪种采样器挑权重、以及单卡还是 4 卡。两者差 0.04–0.06，可看作同一种训练在不同种子、不同挑选方式下的波动幅度。
- λ = 0.1 和 randflip 差不多，比 λ = 0 低约 0.07，落在这个波动幅度附近。
- 采样器的影响约 +0.30，三组一致。
