# 20261118_difusco_rblarge_baseline：RB-[800-1200] 上的 DIFUSCO 基线

给 `20261118_fm_rblarge_mcmcstart`（MCMC 起点的 flow matching / bridge matching）做对照。用户 11-17 夜里定：和 RLSA、DIFUSCO 比；DIFUSCO 用我们之前一直用的 8 层 × 64。

## 配置（除方法外和 FM / BM 两组一致）

- 数据：同一份本地 KaMIS 标签，`/usr0/home/shengyuf/yunyi/data/rb_large_kamis/`，训练 4000 张，验证取前 64 张，测试 1000 张。
- 网络：EncodeProcessDecode，8 层 × 64。
- 训练：均匀前向（D3PM，1000 步，线性），每步 40 张图（2 卡 × 20），AdamW，学习率 2e-4，cosine 衰减，200 个 epoch，不早停，每个 epoch 存权重，activation checkpoint，图缓存。
- 采样：只用 D3PM 后验（用户 11-17 定），验证选 best.pt 和测试都用它。
- 测试：50 步（cosine）× 20 条链，贪心解码，取最好；seed 0、1。

## 流水线

`pipeline.sh`：等 FM 流水线建好图缓存（它先等 KaMIS 标签）→ 从 GPU 9 往下找 2 张空卡训练 → 同两张卡上跑 2 个测试。日志在 `runs/experiments/20261118_difusco_rblarge_baseline/pipeline.log`，wandb 项目 SLVUL，run 名 `20261118_difusco_rblarge_baseline_s0`。

## 结果

（待补）
