# 实验目录

每一次实验一个子文件夹，里面放这次实验的启动脚本和实验记录。从 2026-11-16 起，SL vs UL（DIFUSCO vs DiffUCO，RB-[200-300]）的实验都放在这里。

## 子文件夹命名

`YYYYMMDD_<简短说明>`，日期是开跑那天（UTC），例如 `20261116_rbsmall_difusco_lr`。一次实验可以包含多个 run（不同超参、不同种子），只要它们回答同一个问题。

## 每个子文件夹里放什么

复制 `_template/` 开始：

- `launch.sh`：完整的启动命令，自己 source 环境，不依赖当前 shell。
  - 所有超参都写在命令里，不靠代码默认值（默认值也写出来）。
  - 一个 run 一行调用，或用循环；run 名 = 子文件夹名 + 超参/种子后缀。
- `README.md`：实验记录。包括：
  - 目的：这次实验要回答什么问题；
  - 配置：和基准配置的差别、GPU、等效 batch、预计卡时；
  - 状态：开跑时间、进程号、GPU、完成时间；
  - 结果：验证集和 test 的数字，以及 wandb 链接；
  - 结论。
- `wandb/`：这次实验所有 run 的 wandb 本地记录，由 `launch.sh` 里的 `WANDB_DIR` 指到这里，每个 run 一个 `run-<日期>_<时间>-<id>/`。
  - 进 git 的只有每个 run 的 `files/config.yaml`（全部超参）和 `files/wandb-summary.json`（最后的指标）。
  - 二进制记录 `*.wandb`、`output.log`、debug 日志、`wandb-metadata.json`（含用户名和主机名）留在本地，不进 git（规则在仓库根目录的 `.gitignore`）。
- 小的结果文件（汇总表 CSV、画图脚本）可以放进来；**checkpoint、完整训练日志不进 git**。

## 输出放在哪

- checkpoint、训练日志写到 `runs/experiments/<子文件夹名>/`（`runs/` 已在 `.gitignore` 里）；wandb 记录写到本子文件夹的 `wandb/`。
- wandb：entity `yunyi-organization`，project `SLVUL`。启动前 `export CAM_WANDB=1` 再 source `cam/scripts/env.sh`，key 从仓库外的私有文件读（`/usr0/home/shengyuf/yunyi/.ssh/wandb_api_key`），不要写进脚本或提交。

## 规矩

- 只用 `CUDA_VISIBLE_DEVICES` 指定的空闲卡；`cam_require_gpu` 会检查。
- 选超参、选 checkpoint 只看验证集；test 只在定稿后测。
- 实验跑完、结论写进 README 后再提交；跑之前先提交启动脚本，保证记录和实际运行的命令一致。
