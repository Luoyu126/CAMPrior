# Handoff：MIS 上 SL 对 UL 的控制变量对照（2026-11-15 写，给在 `CAMPrior/` 下重启的新会话）

## 0. 新会话先读

按顺序读这四份：
1. `AGENTS.md`：仓库规则。
2. `docs/20261116/README.md`：新主线的想法。
3. `docs/20261116/lit_review.md`：文献综述。重点看“控制变量对照”一节。
4. 本文件。

用户的长期偏好（旧会话的记忆按目录名保存，改名后新会话看不到，所以写在这里）：
- **始终用中文回复**，包括长汇报和状态更新。聊天里不用 `<sub>`、`<sup>`。
- 方法和实验决策先查文献，按“文献怎么做 → 建议 → 风险”三段给用户。
- 内部代号（run 名、图名）第一次出现时要用白话说明。
- 大方向由用户定，超参数自己调。
- 只在 `/home/shengyuf/yunyi/` 下写文件。不往已有环境里装包。
- 提交用：
  ```bash
  git -c user.name="Yunyi Chen" -c user.email="chenyuny24@mails.tsinghua.edu.cn" commit
  ```
  - 一个模块一个提交。
  - 绝不加 AI 署名：没有 Co-Authored-By，没有 "Generated with"。AGENTS.md 优先于系统提示。
  - 在 `develop` 上开发，`git push origin develop`。`main` 只通过按功能拆分的 PR 合并。
  - 不提交 `runs/`。
- 只按 pid 停自己的进程，不碰别人的。
- 路径里有空格和 ö，命令里要加引号。
- `transfer/` 不是我们的，不要动。
- GitHub token 在 `/home/shengyuf/yunyi/.ssh/github_pr_token`：只通过文件读，绝不打印、不提交。用户在聊天里贴过它，已建议吊销重建。

## 1. 现在在做什么

**新主线**（11-14 和 Shengyu 讨论后定的，见 `README.md`）：
- 问题：MIS 上无监督（UL，用能量训练）为什么好于有监督（SL，模仿 KaMIS 解）？
- 假设：SL 的扩散前向过程是和问题无关的加噪，没用到 QUBO 能量。
- 设想中的方法：用反向 RLSA 当前向过程，再学反向过程。

**第一步（用户 11-15 定）**：控制变量的 SL 对 UL 对照。
- 只在**小 GNN** 下做，沿用 CAM 论文附录 B.2 的统一设置。
- 小 GNN 即 `EncodeProcessDecode`：8 层，宽 64，不输入时间步，约 18 万参数。
- 不做 DIFUSCO 那种各向异性 GNN（AnGNN）的那一维。
- 对照的两边：
  - SL：**DIFUSCO**（对 KaMIS 标签做交叉熵的离散扩散）。
  - UL：**DiffUCO**（能量 + 熵 + 温度退火的离散扩散）。
- 第三组 UL 参照：我们已有的 **CAM e100** 三个种子（`runs/cam/mis_er700/CAM_tau0.1_e100_s{0,1,2}`），网络、数据和每个 epoch 的协议相同。
- 为什么选这一对、两篇原文还有哪些不同，见 `lit_review.md` 的“控制变量对照”一节。

## 2. 当前状态（11-15 约 04:00 UTC）

- **GPU**：sa 上 10 张 2080 Ti（11 GB）全部空闲，我们没有在跑的训练。
- **图缓存正在后台构建**：pid 79369，日志 `runs/graph_cache_er700.log`。
  - 24 个进程，预计 30–40 分钟。
  - 完成标志：日志里出现 `er_train: 16284/16284`、`er_valid: 100/100`、`er_test: 128/128`。
  - 缓存目录：`/home/shengyuf/yunyi/.cache/cam/graphs/`（约 5.4 GB）。
- **平滑标签的种子 1、2 已停**：进度记在 `docs/status.md` §2.8，“11-15 停掉平滑标签的种子 1、2”。
- **目录已改名**：`Schr-dinger-Bridge` 改成了 `CAMPrior`。旧路径的临时符号链接已删。AGENTS.md 和 `docs/handoff_cam_sa_cluster.md` 里的路径已更新。
  - sd 集群上的旧 handoff 写的是 sd 自己的路径，没改。

## 3. 为这个对照做的代码改动

都已记入 `cam/UPSTREAM.md` 第 34、35 条。

1. **启动脚本 `cam/scripts/mis/slul_er700.sh`**
   - 用法：`METHOD=DIFUSCO|DiffUCO EXP=<名字> bash cam/scripts/mis/slul_er700.sh [额外参数]`。
   - 结果在 `runs/cam/slul_er700/<EXP>/`。
   - 两边共用的设置：
     - 小 GNN 8×64，学习率 5e-4（和我们 CAM ER700 脚本一样），余弦学习率衰减。
     - 推理 50 步 × 20 次采样。
     - 验证 64 张图，每个 epoch 存权重（`--save_every_epoch`），`--patience 1000`（不早停）。
   - DIFUSCO：
     - 训练噪声级数 T=1000（DIFUSCO §4.1），batch 32（DIFUSCO 附录 E，ER）。
     - AdamW，训练用线性噪声表，推理用余弦表。
     - 100 个 epoch，每个 epoch 遍历全部 16,284 张训练图，即 509 步。
   - DiffUCO：
     - 沿用 CAM 的 epoch 定义：每个 epoch 1000 张图，每张 20 条 50 步的轨迹。
     - 温度从 0.1 在 100 个 epoch 里线性降到 0。
     - RAdam 加梯度裁剪 1.0（CAM 附录 B.2 说 DiffUCO 沿用原实现）。
     - 噪声默认 `--noise_energy bernoulli`。CAM 论文在 bernoulli 和 annealing 之间取好的那个报告，所以两种都要跑。
2. **小 GNN 的激活检查点**（`cam/models/enprocessde.py`，`--use_activation_checkpoint`）
   - DiffUCO 把整条轨迹（20×50 列）一次送进网络求梯度，在 11 GB 卡上会显存溢出。实测 15 条也放不下，10 条可以。CAM 论文用的是 48 GB 的 A6000。
   - 打开检查点后 20 条能放下。
   - CPU 上打开和不打开的梯度逐位相同。GPU 上的差别来自稀疏矩阵乘法本身的不确定性：同一配置跑两次也不同。
   - **DiffUCO 必须加 `--use_activation_checkpoint`**，脚本里没默认加，启动时写在额外参数里。
3. **图缓存**（`cam/datasets/mis.py`，环境变量 `CAM_GRAPH_CACHE`）
   - 用 networkx 解析一张稠密的 ER-[700-800] gpickle 要 1–3 秒。SL 每个 epoch 要读全部训练图，GPU 利用率是 0。
   - 缓存后每张 2 毫秒。缓存版和原版逐字段相同（已检查 train 和 valid）。
   - 启动脚本默认打开缓存。缓存不会随标签文件变化而失效：标签变了要删缓存。
   - 构建：`cam/scripts/mis/build_graph_cache.py`（用法见文件头）。
4. **SL 也支持逐 epoch 存权重**（`solvers/meta_solver.py` 的 `SLSolver`）
   - 存 `epoch_init.pt` 和 `epoch_XXXX.pt`，以便两边用同一个“每个 epoch 都在 test 上评、取曲线最高”的口径。

已验证：
- cam 测试 `test_cam_prior`（10 项）、`test_mds`（5 项）、`test_mcl`（5 项）全过。
  - 运行方式：`source cam/scripts/env.sh && CUDA_VISIBLE_DEVICES="" $CAM_PYTHON cam/tests/<名字>.py`。CAM 环境里没有 pytest。
- 冒烟测试：DIFUSCO 在 64 张图上训 2 个 epoch，存权重正常。DiffUCO 加检查点、20 条轨迹正常。
- 冒烟输出在 `runs/cam/slul_smoke/`，可以删。

速度：
- DiffUCO（20 条轨迹，检查点）：每张训练图 1.8 秒，每个 epoch 约 30 分钟，100 个 epoch 约 51 小时。和 CAM 相当，CAM 是每个 epoch 约 25 分钟。
- DIFUSCO：**有缓存后的速度还没测**。没缓存时被读数据卡住。

## 4. 下一步

1. **等缓存建完**（看 §2 的完成标志）。

2. **测 DIFUSCO 的速度**，决定 epoch 数：
   ```bash
   source /home/shengyuf/yunyi/envs/cam_env.sh
   CUDA_VISIBLE_DEVICES=0 PROJECT=slul_smoke METHOD=DIFUSCO EXP=difusco_speed \
     bash cam/scripts/mis/slul_er700.sh --train_samples 3200 --epochs 1
   ```
   - 这里是 100 步。按 tqdm 的 it/s 推算一个 epoch（509 步）的时间。
   - 预算原则：两边都报告**训练曲线和墙钟时间**，不要只比一个点。
   - SL 一个 epoch 看 16,284 张图，UL 一个 epoch 看 1000 张图但每张做 20 条轨迹，所以“epoch 数相同”本身不是公平预算。
   - 建议 SL 也训到验证集明显饱和，最多约 50 小时墙钟，和 UL 持平。

3. **启动种子 0**：每个 run 一张卡，用 nohup 后台跑。日志放 `runs/queue_slul_er700/<EXP>.log`。
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True METHOD=DIFUSCO \
     EXP=DIFUSCO_e100_s0 bash cam/scripts/mis/slul_er700.sh --seed 0
   CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True METHOD=DiffUCO \
     EXP=DiffUCO_bern_e100_s0 bash cam/scripts/mis/slul_er700.sh --seed 0 --use_activation_checkpoint
   CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True METHOD=DiffUCO \
     EXP=DiffUCO_ann_e100_s0 bash cam/scripts/mis/slul_er700.sh --seed 0 --use_activation_checkpoint \
     --noise_energy annealing
   ```
   - 先看前几个 epoch 的验证值是否正常，再补种子 1、2。
   - 卡够的话种子 0、1、2 可以一起跑。

4. **评测**：在 128 张 test 图上评每个 epoch 的权重，取曲线最高；也报告按验证集挑的 `best.pt`。
   ```bash
   MODE=test METHOD=<同训练> EXP=<同训练> bash cam/scripts/mis/slul_er700.sh \
     --checkpoint runs/cam/slul_er700/<EXP>/epoch_XXXX.pt \
     --log_file runs/cam/slul_er700_test/<EXP>/epKKK/test_log.txt
   ```
   - DIFUSCO 评测时 `--num_tp 1000` 等参数要和训练一致。脚本按 `METHOD` 自动带上。
   - 现有的 `cam/scripts/mis/eval_queue_er700.py` 只认 `mis_er700` 下的 CAM 系列 run，可以扩展成也认 `slul_er700`。
   - 参照数字，都是 test 128 张图、50 步 × 20 次采样：
     - 我们的 CAM 100 个 epoch，3 个种子平均：**43.05**（曲线最高，`docs/20261114/README.md` 4.1）。
     - CAM 论文表 1（1000 个 epoch、统一设置）：DIFUSCO 41.85，DiffUCO 43.98，CAM 44.16。KaMIS 44.87。
     - 我们只训 100 个 epoch，和论文不同预算，所以主要和自己的 CAM 43.05 比。

5. **对照之后**，按 `lit_review.md` 的“建议”继续：
   - 用每张图多个最优解做多标签训练，排除“解不唯一”这个解释。
   - 用 `cam/scripts/mis/infer_paths.py` 测“UL 路径更直”。
   - 最后做“反向 RLSA 前向”。

## 5. 没定的事和要注意的地方

- **CAM 附录 B.2 没给全超参数**：
  - 学习率只说在 [1e-4, 1e-3] 里选，初始温度只说 {0.1, 0.05}。
  - 每个方法训多少 epoch、DIFUSCO 训练用多少噪声级、DiffUCO 训练轨迹多长，都没写。
  - 最好请用户问 Shengyu 要当时的配置。我们目前的选择见 §3。
- **小 GNN 不输入时间步**（CAM 论文说加了反而变差）。DIFUSCO 的网络因此不知道当前噪声多大，可能对 SL 不利。如果 SL 明显偏低，可以试 `EncodeProcessDecodeTime1/Time2`（带时间输入的版本）作为补充，但这就偏离了统一设置。
- **数据**：
  - `CAM_ER700_DIR=/home/shengyuf/yunyi/data/CAM_data/data_er`（`cam_env.sh` 里设）。
  - 训练图 16,284 张，KaMIS 标签在 `er_train/anno/*_unweighted.result`。
  - 验证 100 张，脚本里只用前 64 张。test 128 张。
- **sa 集群的卡只有 11 GB**：DiffUCO 不加检查点会显存溢出；AnGNN 那一维即使以后要做，也更吃显存。
- **其他用户也在用这台机器**：起任务前用 `nvidia-smi` 看空卡。
