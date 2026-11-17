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
