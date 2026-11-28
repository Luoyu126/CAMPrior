# 路线 1'：DSBM（两个模型交替训练的薛定谔桥），RB-small

依据 `docs/20261128/plan_route1prime.md`（方案），进度 `docs/20261128/status_route1prime.md`。

## 做法

一个 run 是 `cam/scripts/sb/dsbm_loop.py` 的一次调用（`launch.sh <名字> <参数>`），输出在
`runs/experiments/20261128_dsbm_rbsmall/20261128_dsbm_rbsmall_<名字>/`：

- `fwd_it<k>/`、`bwd_it<k>/`：第 k 轮的生成网络 G、加噪网络 N（`train.log`、每个 epoch 的检查点）。
- `fwd_it<k>/anchor_*_result.txt`：第 k 轮 G 最后一个 epoch 在验证锚点（验证图第 64–499 张）上的评测；`eval_done.json` 汇总，
  `start_dependence.txt` 是起点依赖（评测种子 100 的成对追踪）。
- `caches/`：配对缓存 `f<k>.pt`（G 的训练配对）、`b<k>.pt`（N 的训练配对），轨迹 `trace_*_it<k>.jsonl`。
- `state.json`：每轮完成到哪一步、每轮的训练动态指标（`metrics`）、停在第几轮；`loop.log`：过程记录。

两种模式：
- `--mode dsbm`：G、N 每轮各采 16 条链，把配对交给对方（1D 的交叉喂入），各训练 50 个 epoch（第 0 轮 100 个），接着上一轮的参数，
  学习率余弦调度每轮重启。G 的样本贪心解码成合法独立集再交给 N。
- `--mode frozen`：同样的轮数与 epoch，只有 G，每个 batch 用闭式配对（ω > 0：Sinkhorn 势函数）或独立配对（ω = 0）。

第 0 轮 G 从 11-19 的 KaMIS 多解独立配对模型（400 epoch，`20261119_difusco_rbsmall_reach_q5_mkamisfix_long_soft_s0`）出发。

停止规则（方案 §3，统计量见状态文档的偏离记录）：至少 10 轮；之后最近 3 轮验证集前 64 张图 MIS（每轮最后 10 个 epoch 的平均）
每次变化 ≤ 0.03，且两个缓存的平均 Hamming 代价每次相对变化 ≤ 1%，就停；最多 30 轮。同组对照用 `align.sh` 统一到最晚停下的轮数。

## D1：KaMIS 多解，四个 run

```bash
E=experiments/20261128_dsbm_rbsmall
G=/usr0/home/shengyuf/yunyi/data/rb_small_sb/logg_mkamisfix_w0.35_K4194304.pt
bash $E/launch.sh d1_dsbm_w0.35   --mode dsbm   --omega 0.35 --gpus 0,1
bash $E/launch.sh d1_dsbm_w0      --mode dsbm   --omega 0    --gpus 2,3
bash $E/launch.sh d1_frozen_w0.35 --mode frozen --omega 0.35 --gpus 4 --logg $G
bash $E/launch.sh d1_frozen_w0    --mode frozen --omega 0    --gpus 5
nohup bash $E/align.sh d1_dsbm_w0.35 d1_dsbm_w0 d1_frozen_w0.35 d1_frozen_w0 > runs/experiments/20261128_dsbm_rbsmall/align_d1.log 2>&1 &
```

结果：见状态文档与总报告。
