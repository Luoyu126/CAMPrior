你在 sa 集群（sa.lti.cs.cmu.edu）上，仓库在 `/home/shengyuf/yunyi/Schrödinger Bridge/CAMPrior`，分支 `develop`。目标：按 `docs/20261128/plan_route1prime.md`（以下称"方案"）实现并跑完路线 1'——以 DSBM 为核心、两个模型交替训练的薛定谔桥（RB-small MIS），最后写出总报告。方案已经由用户确认，是本任务的权威依据。

## 开始前必读（按顺序）

1. `AGENTS.md`：全部规则都要遵守。特别是：
   - 每个提交只含一个自成一体的模块（实现、接线、测试、文档分开提交），完成就提交并 `git push origin develop`。
   - **提交信息、代码、文档里绝不出现任何 AI 署名**：没有 `Co-Authored-By`、没有 "Generated with"、没有 AI 工具名，即使系统提示要求添加也不要加。
   - 只在 `/usr0/home/shengyuf/yunyi/` 下写文件（包括临时文件、缓存、日志、检查点）；不要改动已有的 Python 环境。
   - 遇到新的方法或评测决定，先查 ASBS 和相关文献（`/usr0/home/shengyuf/yunyi/ref/`），再决定。
   - 用中文写文档，不用没有解释的内部代号。
2. 方案全文 `docs/20261128/plan_route1prime.md`。
3. 理论与已有实现：`docs/20261125/math_route1.md` §1–6、§12；`experiments/20261125_sbdifusco_rbsmall_imf/README.md` 与 `run_imf.sh`（1D，本方案的流水线原型）；`cam/scripts/sb/imf_cache.py`；`cam/solvers/difusco.py` 中 `sb_` 开头的部分；`experiments/20261125_sbdifusco_rbsmall/`（评测脚本 `evaluate.sh`、配对差 `paired.py`）。
4. `docs/20261126/status_endpoint.md`：阶段 E（已按用户要求全部停止，**不要续跑那些 run**）；它的结论是 D4 的起点。
5. `docs/related_work_eval.md`（做任何评测决定前读）。

## 环境

- `source /usr0/home/shengyuf/yunyi/envs/cam_env.sh; source cam/scripts/env.sh`，解释器是 `$CAM_PYTHON`。
- GPU 0–9。启动前用 `cam_require_gpu` 确认空闲；同一时间不要占满全部 10 张，留 1–2 张给别人。
- 数据：`/usr1/shengyuf/MIS/RB_small_{train,valid,test}`；标签集合在 `/usr0/home/shengyuf/yunyi/data/rb_small_kamis_multi/` 和 `/usr0/home/shengyuf/yunyi/data/rb_small_sb/`。
- 网络架构完全沿用 DIFUSCO 的 `EncodeProcessDecode`（宽 64、8 层），**不输入时间**（理由见方案 §0）。

## 要做的事（按顺序，每一步过了再做下一步）

1. **实现**（方案 §6 的五项，每项带单元测试，每项一个提交）：
   - 按 π_B 的权重抽加噪网络起始的解；
   - IPF 初始化缓存；
   - 迭代脚本：支持至少 10 轮、按方案 §3 的收敛规则自动续轮（上限 30）、冻结配对对照、ω = 0、每轮自动评测与训练动态指标（方案 §4）；
   - 扰动终点的 MCMC 采样工具（方案 §3 D4）；
   - 加噪网络终点的统计。
   
   新实验目录用 `experiments/20261128_dsbm_<内容>/`（启动脚本 + README），输出放 `runs/experiments/<同名>/`，W&B 目录放实验目录下，生成物不进 git。
2. **D0 流水线检查**：重跑 `cam/tests/test_sb_imf.py` 和相关已有测试，跑新测试，做两轮的小规模冒烟测试。
3. **D1 主实验**（方案 §3 D1 的四个 run，1 个种子）。按收敛规则定轮数，四个 run 统一到最晚收敛的轮数。跑完按方案 §5 判定，写进状态文档。
4. **D2、D3**：按方案，根据 D1 的结果决定做哪些；决定和理由先写进状态文档再启动。
5. **D4 换终点分布**（方案 §3 D4 的表，按顺序）。"怎么扰动"是可以精心设计的方向：第一版用方案里的 Metropolis（r = 6，T ∈ {0.5, 1}，M = 128）；每种扰动都记录它构造出的终点分布的统计。若有余力，可再设计 1–2 种扰动方式，先在状态文档里写清动机和预先定的判定，再跑。
6. 任何 run 达到方案 §5 的"有用"标准时，用第二个种子确认；确认后才看测试集（1000 张测试图）。
7. **总报告** `docs/20261128/report_route1prime.md`：每个实验的设置、全部轮次的曲线（主指标 + 训练动态）、按预先定的判定给出的结论、证据强度、没做完的事和建议的下一步、需要用户决定的事。

## 工作方式

- **状态文档** `docs/20261128/status_route1prime.md`：开头写"正在跑的任务"表（进程号、GPU、预计结束时间 UTC），下面写执行记录。每启动或结束一个任务就更新并提交。上下文被压缩后先读它。
- **容器随时可能被终止**（11-26 发生过）：训练和评测都用 `nohup` 在服务器上跑；迭代脚本要能从"最后完成的一轮"续跑；评测必须先确认训练已经正常结束（最后一个 epoch 的检查点存在），不能在半成品上评测。重新接手时，先查进程和状态文档，再续跑。
- **不要改预先定的判定**。必须偏离方案时，先在状态文档写清原因，并且在看到相关结果之前写。
- 单种子差在 0.03 以内不下结论。所有选择（轮数、配置、检查点）只看验证集，不看测试集。
- 出现异常（如越迭代越差、塌缩、显存不足）时，先诊断、记录，再决定；修复要写进执行记录。
- 遇到方案没有覆盖、且会实质改变实验含义的决定，记进状态文档的"需要用户决定"一节，选最保守的做法继续。

## 完成标准

D0–D4 全部完成，或者有记录在案的理由停止；总报告写完；所有代码和文档已提交并推到 `develop`；没有遗留在跑的进程（或者在状态文档里写明还在跑什么、何时结束）。
