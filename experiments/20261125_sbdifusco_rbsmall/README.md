# 20261125_sbdifusco_rbsmall：路线 1，薛定谔桥配对的 DIFUSCO（RB-small MIS）

依据：`docs/20261125/plan.md` 阶段 1B–1F；数学：`docs/20261125/math_route1.md`；进度：`docs/20261125/status.md`。

## 目的

在 KaMIS 多解标签（`matched_kamis_loaderorder.pt`，每图约 42 个）上，比较独立配对（基线）与薛定谔桥配对（起点 Bernoulli(0.5)^n，"就近"配对，强度 ω）训练出的 DIFUSCO。

## 配置

与基线 `20261119_difusco_rbsmall_reach` 的 `q5_mkamisfix_long{,_soft}` 相同：50 个扩散步、余弦噪声表、400 epoch、每步 40 张图、AdamW 2e-4、checkpoint 按前 64 张验证图选。

- `launch.sh <gpu> <mode> [seed]`：训练。
- `evaluate.sh <gpu> <run dir> <anchor|test> <sampler> <steps> <seed>`：评测。`anchor` 是验证锚点（验证图按加载顺序的第 64–499 张，共 436 张，`/usr0/home/shengyuf/yunyi/data/rb_small_valanchor` 是指向原文件的符号链接）。选配置只看锚点。

## 状态与结果

（进行中，见下文各节。）
