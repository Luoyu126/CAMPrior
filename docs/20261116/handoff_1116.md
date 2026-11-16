# Handoff：SL（DIFUSCO）对 UL（DiffUCO），RB-small 上的严格对照（2026-11-16 00:30 UTC 写）

接着 `handoff.md`（11-15）。用户偏好和仓库规则见那份的 §0 和 `AGENTS.md`：
- 中文回复，内部代号要用白话解释；
- 只在 `/home/shengyuf/yunyi/` 下写文件；
- 提交用 Yunyi Chen 的身份，一个功能模块一个提交，推到 `develop`，不加 AI 署名；
- 只按 pid 杀自己的进程，不要用 `pkill -f`。

## 1. 研究问题和第一步

- **问题**：在 MIS 上，为什么 UL 好于 SL？
- **第一步**：在同一个小 GNN（8 层 × 64，`EncodeProcessDecode`）上，把 SL 和 UL 各自调到收敛，得到稳定的优劣判断。用户要求“非常严谨”：
  - 两边调参的组数相同，卡时对齐；
  - 只按验证集选超参和权重；
  - SL 加算力时要同时加数据。
- **只看 DIFUSCO 对 DiffUCO**，CAM 和 CAMPrior 先放一边。

## 2. 11-15 到 11-16 定下来的事（用户决定）

1. **对照放到 RB-[200-300]（RB-small）上做**，不再用 ER-[700-800]。原因：
   - 图小，每张图训练快约 9 倍；
   - CAM 表 1 里 UL 在这里领先更多：DIFUSCO 18.74，DiffUCO 19.88。
2. **DiffUCO 不调参、不复现**：直接用 CAM 论文那次的记录，已导入两个实验文件夹：
   - `experiments/20260513_diffuco_rbsmall_annealing/`：表 1 用的那次。验证最好 20.27（第 137 个 epoch），test 19.876。
   - `experiments/20260513_diffuco_rbsmall_bernoulli/`：同一天只换了噪声。验证最好 20.11（第 101 个 epoch），test 19.70 / 19.74。
   - 两次的配置：annealing 或 bernoulli 噪声，训练 6 步，每张图 10 条轨迹，β 1.2，τ 从 0.1 在 1000 个 epoch 内退到 0，RAdam，学习率 1e-3，cosine 衰减，梯度裁剪 1。每 epoch 4000 张图，每次更新 40 张图（每卡 4 张 × 10 卡），patience 200。
   - 两次都是早停选出的权重，温度只退到约 0.09，不算训到收敛。
   - 权重在 `runs/experiments/<同名>/best.pt`。
3. **DIFUSCO 由我们在 RB-small 上自己调**。表 1 那次 DIFUSCO 在 sa 上找不到记录，而且 11-07 之前 RB 的标签全是错位的（见 §4），所以 18.74 只作参考。
4. **DIFUSCO 每步 40 张图**，和 DiffUCO 对齐。单卡直接用 batch 40；GraphNorm 按单张图归一化，所以和多卡分摊等价。
5. **ER 上的任务全部停掉**，包括打标签、建缓存、DIFUSCO 1000 epoch、旧的 DiffUCO、两个评测队列。11-16 00:10 起 10 张卡全空。ER 的标签和缓存留在磁盘上。相关提醒已关闭。
6. **命名规则**：`YYYYMMDD_<方法>_<数据集>_<变体>`，例如 `20260513_diffuco_rbsmall_annealing`。wandb run 和本地文件夹同名。

## 3. 基础设施（11-16 新建）

- **实验目录 `experiments/`**：每次实验一个子文件夹，从 `_template/` 复制，放 `launch.sh`、`README.md`、`wandb/`。规则见 `experiments/README.md`。
  - checkpoint 和完整日志放 `runs/experiments/<同名>/`，不进 git。
  - 提交前要删掉 `wandb/*/files/config.yaml` 里的 `_wandb:` 段：里面有邮箱、主机名、GPU 序列号，`wandb sync` 还会把它重新写回来。
- **wandb**：
  - entity `yunyi-organization`，project `SLVUL`，服务器是 forge.coreweave.com。
  - key 存在 `/usr0/home/shengyuf/yunyi/.ssh/wandb_api_key`（权限 600）。不要打印，不要提交。
  - 启动前 `export CAM_WANDB=1`，再 `source cam/scripts/env.sh`；模板里已写好，`WANDB_DIR` 指向实验文件夹。
  - 现在项目里只有两个导入的 DiffUCO run。
- **正号的 MIS 大小**：验证时记录 `eval/mis_size`，test 结果文件多一行 `Avg. size`（`cam/UPSTREAM.md` 第 41 条）。`eval/object` 和 `Avg. obj` 没变，选 best.pt 和各个解析脚本仍用它们。
- **`cam/scripts/mis/collect_wandb_runs.py`**：把一个 wandb 目录整理成 CSV。超参以启动命令为准，`config.yaml` 里的 method 字段有时是错的。

## 4. 代码修复（从上游同步）

- **第 39 条，标签错位**（`cam/datasets/mis.py`）：RB 图的节点插入顺序和编号顺序不一致，旧代码加载的标签 96% 不是独立集。ER 不受影响。
- **第 40 条，DIFUSCO 训练时间步**（`cam/solvers/difusco.py`）：改成从 1..T 抽，不再从 0..T 抽。
- Shengyu 的私有仓库 RiemannCO 里还有一个 FM 推理步的修复，我们没同步，以后用 FM 时再加。

## 5. 我们代码里 DIFUSCO 怎么训练（11-16 和用户过了一遍）

- **加噪**：每张图随机抽 t ∈ 1..1000，按均匀翻转核加噪，β 从 1e-4 线性增加到 0.02。t 超过约 500 后，x_t 基本就是均匀噪声。
- **网络**：输入每个节点当前的 0/1 值，**不输入 t**（`forward` 收到 `time` 但不用），输出 2 类 logits。这是 x₀ 参数化：网络预测干净标签。
- **损失**：和 KaMIS 标签 x₀ 做交叉熵，在所有节点上平均，两类不加权（1 只占约 8%）。
- **优化**：AdamW，weight decay 1e-4，学习率 cosine 衰减到 0，不做梯度裁剪，fp32。每个 epoch 用验证集推理，按结果存 best.pt，patience 早停。
- **推理**：
  - 20 条链从均匀随机出发，按 cosine 时间表走 50 步；
  - 每步：网络预测 x₀，再用 D3PM 后验 q(x_{t2} | x_{t1}, x₀) 采样下一步，下一步重新预测 x₀；
  - 最后一步输出概率，贪心解码成独立集，20 条链取最大。
- **和 DiffUCO 的本质区别**：DiffUCO 的网络直接输出每一步的翻转概率；DIFUSCO 的网络只猜终点，每步怎么走由一个与问题无关的固定后验决定。这一点可以作为以后的机制分析变量。

## 6. 下一步

1. **列 RB-small 上 DIFUSCO 的调参网格和卡时预算，给用户确认后再跑**。
   - 中心配置：每步 40 张，学习率 2e-4，weight decay 1e-4，最多 200 个 epoch，patience 20，`num_tp` 1000，cosine 推理 50 步 × 20 次，种子 0。
   - 参考 Shengyu 的三份脚本，见 `docs/20261116/experiment_plan.md` §6.3：
     - 06-06 版：每步 40–50 张，200 个 epoch；
     - 04-18 版：等效每步 128 张，50 个 epoch。
   - 预算按 DiffUCO 出表那次算：到最好约 11 卡时，到早停约 26 卡时。
   - 还要定：调哪些维度（学习率？加噪表？），每组几个种子，SL 要不要加数据（RB-small 只有 4000 张训练图；原则是 SL 加算力时要同时加数据）。
2. 建实验文件夹（例如 `experiments/20261116_difusco_rbsmall_<变体>/`），从 `_template/launch.sh` 改。
   - 模板现在只写了数据路径和 `--do_train`，要把 DIFUSCO 的全部超参写上：`--method DIFUSCO --model EncodeProcessDecode --num_tp 1000 --num_h 64 --num_l 8 --batch_size 40 --lr ... --weight_decay 1e-4 --epochs ... --patience ... --num_t 50 --num_k 20 --inference_schedule cosine --valid_sample 64`。
   - 验证集张数要和 DiffUCO 对齐：DiffUCO 是 64 张、实际 70 张。单卡跑会正好是 64 张，这个差别要记下来。
3. **待问 Shengyu**：表 1 的 DIFUSCO 是哪天、用哪份代码、在哪台机器上训的（sa 上没有记录；11-07 的批量实验可能在 sf 上）。

## 7. 文档位置

- `docs/20261116/experiment_plan.md` §6.1–6.4：CAM 论文配置核对、改到 RB-small、GPU 配置、DIFUSCO 启动脚本逐项核对、RB 标签错位的影响、11-16 的状态。
  - 注意：§2–§5 还是按 ER 写的旧网格，没有按 RB-small 重排。
- `docs/20261116/cam_paper_diffuco_runs.md`：CAM 论文全部 DiffUCO 运行记录（ER 和 RB），§3.1 是 RB-small 那次的完整配置。
- `docs/20261116/data/cam_paper_diffuco_runs.csv`：38 个 DiffUCO run 的表。
- `docs/20261116/slul_results.md`：ER 上的旧结果和方法讨论（§3 收敛口径，§4 调参原则，§5 两种噪声）。
- Shengyu 的原始记录（只读）：
  - wandb 本地目录：`/home/shengyuf/CAM4CO/wandb/`
  - 输出目录：`/usr1/shengyuf/RiemannCO/`
  - 数据：`/usr1/shengyuf/MIS/RB_small_{train,valid,test}`（4000 / 500 / 1000 张）
- 不要使用 Shengyu `config.sh` 里明文写着的 wandb key。
