# 20261116_difusco_rbsmall_randflip：RB-[200-300] 上的 DIFUSCO（前向过程是随机翻转）

## 目的

SL 一侧在 RB-small 上的第一次训练，和 CAM 论文那次 DiffUCO（`20260513_diffuco_rbsmall_annealing`，验证最好 20.27，test 19.88）对照。先用中心配置跑一次，看 DIFUSCO 在修好标签后能到多少，再定调参网格。

变体名 `randflip`：前向加噪是 D3PM 的均匀随机翻转，Q_t = (1−β_t)I + β_t/2·11ᵀ，β_t 从 1e-4 线性增加到 0.02，共 1000 步。

## 配置

- 启动命令全部写在 `launch.sh` 里，没有依赖代码默认值的超参。
- 网络和 DiffUCO 相同：`EncodeProcessDecode`，8 层 × 64。
- 训练：
  - 每步 40 张图，单卡。DiffUCO 是每卡 4 张 × 10 卡，GraphNorm 按单张图归一化，所以两者等价。
  - 一个 epoch 等于全部 4000 张训练图，即 100 步，和 DiffUCO 的 `samples_per_epoch 4000` 一样。
  - AdamW，学习率 2e-4，weight decay 1e-4，cosine 衰减到 0，不做梯度裁剪，fp32。
  - 最多 200 个 epoch，patience 20。
  - 以上取自 Shengyu 的 `mis/mis_rb_small_difusco.sh`。和那份不同的只有每步张数：那份是 5 × 10 卡 = 50，这里取 40，跟 DiffUCO 对齐。
- 推理和验证：cosine 时间表 50 步，每张图 20 个样本取最好。和 DiffUCO 相同。
- 验证集：64 张，单卡正好 64 张。DiffUCO 用 10 卡，验证补齐到 70 张，这个差别记一下。
- 代码：仓库 `develop` 分支，已包含 RB 标签对齐修复和训练时间步从 1..T 抽的修复（`cam/UPSTREAM.md` 第 39、40 条）。
- 时间：每个 epoch 训练约 2 分钟，验证约 26 秒。跑满 200 个 epoch 约 8 卡时，早停会更短。
- 启动：`bash experiments/20261116_difusco_rbsmall_randflip/launch.sh <gpu> <seed>`

## 状态

| run | GPU | pid | 开跑（UTC） | 结束 |
|---|---|---|---|---|
| `20261116_difusco_rbsmall_randflip_s0`（wandb `rox3ivnr`） | 0 | 2646007 | 2026-11-16 00:36 | 跑着 |

- 11-16 02:10（UTC）检查：跑完 37 个 epoch（第 0–36 个），没有早停，没有报错。
  - 验证最好 19.27，在第 28 个 epoch（`best.pt` 文件时间 01:47，对得上）；之后第 30、36 个 epoch 是 19.19、19.20。
  - 曲线：第 0–8 个 epoch 在 17.8–17.9 不动，第 9 个 epoch 起上升，第 22 个 epoch 起到 18.9–19.3。
  - 速度：37 个 epoch 用了 94 分钟，每个 epoch 约 2.5 分钟。跑满 200 个 epoch 约在 11-16 09:05（UTC）结束；若第 29–48 个 epoch 都不超过 19.27，会在第 48 个 epoch 后早停，约 02:40（UTC）。
  - 期间 wandb 上传偶尔报 Service Unavailable 后自动重试成功，不影响训练。
- 11-16 02:52（UTC）检查：跑完 54 个 epoch（第 0–53 个），没有早停。
  - 验证最好 19.45，在第 47 个 epoch（`best.pt` 文件时间 02:35）；第 52 个 epoch 19.44，第 44、49 个 epoch 19.41。
  - 第 36–53 个 epoch 在 19.03–19.45 之间，还在缓慢上升。
  - 每个 epoch 约 2.5 分钟。最早在第 67 个 epoch 后早停，约 03:27（UTC）；跑满 200 个 epoch 约 09:00（UTC）。
- 11-16 04:08（UTC）检查：跑完 82 个 epoch（第 0–81 个），没有早停。
  - 验证最好 19.56，在第 66 个 epoch（`best.pt` 文件时间 03:22）；第 72 个 epoch 19.55，第 77 个 epoch 19.52。
  - 第 54–81 个 epoch 在 19.23–19.56 之间，涨得已经很慢。
  - 04:05 前后机器上别的用户开了几十个 CPU 进程（负载 79，共 80 核），GPU 利用率掉到 11%，验证从每张 0.37 秒变慢到约 1.7 秒（推理结果不受影响，只是变慢）。
  - 最早在第 86 个 epoch 后早停；跑满 200 个 epoch 视机器负载约在 09:00–12:00（UTC）结束。
- 11-16 05:40（UTC）检查：跑完 95 个 epoch（第 0–94 个），没有早停。
  - 验证最好 19.59，在第 83 个 epoch（`best.pt` 文件时间 04:17）；第 66、86 个 epoch 19.56。
  - 第 82–94 个 epoch 在 19.39–19.59 之间，基本走平。
  - 别的用户的 CPU 进程还在（负载约 79），验证每张约 1.4 秒，每个 epoch 约 7 分钟，比开头慢近 3 倍。
  - 最早在第 103 个 epoch 后早停，约 06:45（UTC）；照现在的速度跑满 200 个 epoch 要到约 18:00（UTC）。

## 结果

| run | 验证最好（epoch） | test | wandb |
|---|---|---|---|

## 结论
