# SELF_CHECK — 阶段 3e：官方 SuGaR dn_consistency 对照组 `coarse_official_dnc`

> 执行者 8（opus-executor）。本文件在运行过程中分段写入，**最终状态见文末「12. 总结论」**。
> 硬性规则遵守情况：未删除任何文件、**我本人未执行任何 `git commit` / `git add`**、未 `pip install`、
> 未修改任何已有文件（只新增 4 个文件）、等到前序任务发出信号后才启动，**启动时独占 GPU0**；
> 训练末段与 mesh 段被他人后起的作业并发占卡（见 §11 偏离 2，只影响耗时不影响精度/几何指标）。

---

## 1. 目的

官方 SuGaR 仓库自带一份 2DGS 式的深度-法向一致性正则
（`sugar_trainers/coarse_density_and_dn_consistency.py`），与本项目自研的 DNC（`sugar_utils/dnc_utils.py` +
`sugar_trainers/coarse_density.py` 补丁）思路重合。官方版本只能通过 `train.py -r dn_consistency` 调用，
而那是「coarse + refine + 贴图网格」的全流程脚本，本次考核不跑 refine，因此**需要一个只跑 coarse 的入口**，
才能在同一场景、同一 3DGS 7k 权重、同一迭代预算、同一评测协议下与自研 DNC 正面对比。

---

## 2. 新增文件（只增不改）

| 文件 | 作用 | md5 |
|---|---|---|
| `repo/SuGaR_dev/train_coarse_official_dnc.py` | 官方 dn_consistency coarse 训练入口（本次实际运行） | `ff04346a57b2f6be54be980956c79478` |
| `repo/SuGaR/train_coarse_official_dnc.py` | 同一文件的副本（分支 `dnc`，留作仓库交付用） | `ff04346a57b2f6be54be980956c79478` |
| `scripts/run_s3e_official_dnc.sh` | 等待 GPU 信号 → 训练 → mesh 提取 → 评测的串联脚本 | — |
| `scripts/eval/blocktime_from_logs.py` | 从训练日志重算每迭代耗时并剔除被 GPU 争用的分块（§9.1 的数字来源） | — |

启动前后两个仓库的 `git status --short`（证明没有改动任何已有文件、也没有 commit）：

```
# repo/SuGaR (branch dnc)，我新增文件时（08:31）的状态：
 M sugar_trainers/coarse_density.py      <- 阶段 3 的 DNC 补丁（他人所为，我未触碰）
 M train_coarse_density.py               <- 同上
?? sugar_utils/dnc_utils.py              <- 同上
?? train_coarse_official_dnc.py          <- 本次新增

# repo/SuGaR_dev (branch main @ 7c10c4a)，本文件写完时仍是：
 M extract_mesh.py                       <- 他人 09:08 的 Frosting 改动（见偏离 3）
 M sugar_extractors/coarse_mesh.py       <- 同上
 M sugar_trainers/coarse_density.py      <- 阶段 3 的 DNC 补丁（他人）
 A sugar_utils/dnc_utils.py              <- 同上
 M train_coarse_density.py               <- 同上
?? train_coarse_official_dnc.py          <- 本次新增
```

> ⚠️ **更正（09:26 复核时发现）**：`repo/SuGaR` 现在是干净的，因为**主会话（Claude Fable 5.1）于
> 09:19:16 做了 commit `39c0f56`「中期快照」**，把我新增的 `train_coarse_official_dnc.py`
> （commit 内 md5 仍是 `ff04346a57b2f6be54be980956c79478`，与我写的逐位一致）和
> `scripts/run_s3e_official_dnc.sh` 一并收录。**这次 commit 不是我做的**——我全程没有执行过任何
> `git commit` / `git add`。`repo/SuGaR_dev` 未被 commit，HEAD 仍是 `7c10c4a`。

`sugar_trainers/coarse_density_and_dn_consistency.py` 在两个仓库里 `diff` 结果为**完全一致且未被修改**
（不在 `git status` 中出现），即官方正则代码一行未动。

---

## 3. 官方 DNC 与自研 DNC 的实现差异（用于解释指标差异）

| 维度 | 官方 `coarse_density_and_dn_consistency.py` | 自研 `dnc_utils.py` + `coarse_density.py` |
|---|---|---|
| 损失式 | `mean(1 - <N_view , N_d_view>)`，**全图像素求平均** | `mean(1 - \|<N, N_d>\|)`，**只在有效像素上平均** |
| 绝对值 | 无（依赖法向已被显式翻向相机） | 有（消除 3D 椭球最短轴的朝向歧义） |
| 有效像素掩码 | **无**（背景/未命中像素也参与） | `D < 0.98·max_depth`、去 2 像素边界、深度相对梯度 < 0.05、`‖N_raw‖ > 0.1` |
| 法向来源 | `sugar.render_depth_and_normal()`：逐高斯最短轴 → 按 `sign((n·(c−p)))` 翻向相机 → 渲染 view 空间 x,y 分量 → `z = −sqrt(1−x²−y²)` | 逐高斯最短轴 → 翻向相机 → 渲染三分量 → 像素上再归一化 |
| 深度→法向 | 2DGS 原版 `depths_to_points` + `depth2normal_2dgs`（COLMAP 相机、world 空间叉乘后转 view 空间），中心差分 | PyTorch3D NDC 解析反投影 + 中心差分叉乘，全程 view 空间 |
| 梯度路径 | 同时经深度 D 与法向 N（无 detach） | 默认同官方；`--dnc_detach_depth True` 时只经 N（阶段 3d 新增） |
| 权重 λ | `dn_consistency_factor = 0.05`（源码硬编码） | 命令行 `--dnc_factor`，本次对照取 0 / 0.05 / 0.2 |
| 起始迭代 | `start_dn_consistency_from = 9000`（硬编码），条件 `iteration > 9000` | `--dnc_start 9000`，同条件 |

> 结论性提示：**官方版本的 λ=0.05 与自研 `coarse_dnc005` 是最接近的一对**，两者主要差在
> 「有无有效像素掩码」「有无绝对值」「深度→法向的具体离散化」三点。

---

## 4. 新入口脚本做了什么 / 没做什么

`train_coarse_official_dnc.py` 是**薄封装**：

- 参数与 `train_coarse_density.py` 完全一致（`-s -c -o -i --eval --white_background -e -n --gpu`），另加 `--seed`（默认 0）。
- 调用前 `random.seed / np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all`（与自研训练器同一写法、同一时机）。
- 用 `time.time()` 与 `torch.cuda.max_memory_allocated()` 在调用前后测墙钟与峰值显存（调用前先 `reset_peak_memory_stats`）。
- 训练结束后从返回的 `model_path`（`.pt`）读 `state_dict['_points'].shape[0]` 统计高斯数。
- 写 `<output_dir>/train_stats.json`，字段名与自研 run 对齐（`train_wallclock_min`、`mean_time_per_iteration_ms`、
  `max_memory_allocated_MiB`、`n_gaussians_final`、`seed`），另加 `regularization='official_dn_consistency'`、
  `dn_consistency_factor=0.05`、`start=9000`。
- **不改官方训练器一行代码**，不改任何超参（λ 与起始迭代都由官方源码硬编码）。

---

## 5. CPU 侧自检（GPU 空出前完成）

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法编译（dev） | `python -m py_compile train_coarse_official_dnc.py`（在 `repo/SuGaR_dev`） | PASS |
| 语法编译（SuGaR） | 同上（在 `repo/SuGaR`） | PASS |
| `--help` 真导入 | `python train_coarse_official_dnc.py --help` | PASS，退出码 0，10 个参数齐全 |
| argparse 解析值 | 打桩脚本（stub 掉官方训练器与全部 CUDA 调用，**不创建 GPU 上下文**）跑真实命令行 | PASS：`ARGPARSE_TEST_OK`，10 个参数逐个与预期一致，`-c` 以 `/` 结尾 |
| 高斯计数函数 | 对已有 `coarse_base_seed0/15000.pt` 调 `_count_gaussians` | PASS：得 429423，与该 run 的 `train_stats.json` 的 `n_gaussians_final` 逐位一致 |
| 串联脚本语法 | `bash -n scripts/run_s3e_official_dnc.sh` | PASS |
| 等待逻辑（超时分支） | `DEADLINE=0001 bash scripts/run_s3e_official_dnc.sh` 干跑 | PASS：打印 "DEADLINE 0001 reached … NOT launching"，退出码 3，**未碰 GPU** |
| 信号 grep 逻辑 | 用 `run_s34_one.sh` 的原样格式伪造一行到临时文件 | PASS：`SIGNAL_GREP_OK` |

打桩脚本：`<scratchpad>/test_parse_official_dnc.py`（不落入项目目录，属临时验证物）。

> lint 说明：本 venv 没有 `pyflakes` / `ruff` / `flake8`，而硬性规则禁止 `pip install`，所以静态检查只做到 `python -m py_compile`（4 个新文件全部通过）+ `bash -n`（脚本通过）。没有装任何新包，`pip list` 未变。

---

## 6. GPU 排队与实际启动

只有一张 H200。按主会话要求，脚本先阻塞等待执行者 7 的 `coarse_dnc02_detach` 链打印
`MESH (coarse_dnc02_detach) finished`（在 `logs/s34_coarse_dnc02_detach.log` 或 `logs/s34_mesh_dnc02_detach.log`），
截止 09:35；超时则**不启动**并退出码 3。等待期间额外确认本用户没有 `train_coarse_density.py` /
`extract_mesh.py` / `train_coarse_official_dnc.py` 进程仍在跑，确保不与他人并发占卡。

- 脚本启动（进入等待）：`2026-09-13 08:34:34`，PID 1339868，等待日志 `logs/s3e_official_dnc_waiter.log`。

（时间线与结果见下节，运行结束后填写。）

---

## 6.1 实际时间线（全部逐字来自日志）

| 事件 | 时刻 | 证据 |
|---|---|---|
| 等待脚本启动（进入轮询） | 08:34:34 | `logs/s3e_official_dnc_waiter.log` |
| 检测到信号 `MESH (coarse_dnc02_detach) finished` | 08:58:55 | 同上（信号写在 `logs/s34_coarse_dnc02_detach.log`，时间戳 08:58:41） |
| 训练开始 | 08:58:55 | `logs/s34_coarse_official_dnc.log` 首行 |
| 官方 DNC 正则启动（iter 9001） | — | 日志第 303 行 `Starting depth-normal consistency.` |
| 训练结束 exit=0 | 09:13:35 | 同上尾行，`wallclock=880s` |
| mesh 提取开始 | 09:13:35 | `logs/s34_mesh_official_dnc.log` |
| mesh 提取结束 exit=0 | 09:18:52 | 同上，`wallclock=317s` |
| 评测（S4.1–S4.4）结束 exit=0 | 09:19:39 | `logs/eval_coarse_official_dnc_091852.log` |

完整命令（逐字，可直接复算）：

```bash
source /scratch/e1351071/zju_test/env.sh
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /scratch/e1351071/zju_test/repo/SuGaR_dev

python train_coarse_official_dnc.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_official_dnc \
  --eval True --gpu 0 --seed 0

python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_official_dnc/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_official_dnc

cd /scratch/e1351071/zju_test
bash scripts/eval/run_eval_for_run.sh coarse_official_dnc \
  /scratch/e1351071/zju_test/outputs/runs/coarse_official_dnc/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  /scratch/e1351071/zju_test/outputs/runs/mesh_official_dnc/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply 0
python scripts/eval/summarize.py
```

（实际由 `bash scripts/run_s3e_official_dnc.sh` 串起来跑，脚本把上面三条命令原样展开并打进日志头。）

---

## 7. 结果 1：训练统计 `outputs/runs/coarse_official_dnc/train_stats.json`

| 字段 | 值 |
|---|---|
| `regularization` | `official_dn_consistency` |
| `dn_consistency_factor` | 0.05 |
| `start` | 9000 |
| `seed` | 0 |
| `train_wallclock_min` | **14.5529** |
| `mean_time_per_iteration_ms`（按 8000 iter 算） | **109.147** |
| `max_memory_allocated_MiB` | **7834.376**（= 7.651 GB） |
| `n_gaussians_final` | **429411** |
| `num_iterations` / `iteration_to_load` | 15000 / 7000 |
| `git_commit` | `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44` |
| 训练末端总损失 | 0.095709（日志 `Training finished after 15000 iterations with loss=0.09570945054292679.`） |

9000 iter 硬剪枝后剩 `429411` 个高斯（日志 `Pruning finished: 429411 gaussians left.`），与 `n_gaussians_final` 一致。

> ⚠️ 口径提醒：本 run 的 `train_wallclock_s` 是**整个训练函数**（含数据加载 ≈ 35 s）的墙钟，
> 而自研 run 的 `train_wallclock_s` 从训练循环开始计时。用官方训练器自己打印的
> 「computed in X minutes」逐块求和得到**同口径的循环墙钟 = 13.976 min（104.82 ms/iter）**。
> 该求和法在 `coarse_dnc005` 上验证过：求和 13.9179 min vs 其 `train_stats.json` 的 13.93 min，
> 偏差 < 0.1%。复算命令见 §9。

## 8. 结果 2：mesh 提取

| 项 | 值 |
|---|---|
| 输出 | `outputs/runs/mesh_official_dnc/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| 文件大小 | 15,278,291 字节 |
| 顶点数 V | **205,113** |
| 面数 F | **370,554** |
| Poisson depth / cleaning quantile | **10 / 0.1**（与其余各组完全一致，见 `mesh_official_dnc/extract_stats.json`） |
| 墙钟 | 317 s（**与他人的 2 个 `extract_mesh.py` 并发，见 §10 偏离 2**） |

## 9. 结果 3：五组并排（全部数字逐字来自 `outputs/metrics/summary.csv`）

| 指标 | base_seed0 | dnc005 | dnc02 | dnc02_detach | official_dnc |
|---|---|---|---|---|---|
| PSNR (dB) ↑ | 24.6201 | 22.4473 | 13.2436 | 24.3995 | 24.3779 |
| SSIM ↑ | 0.85552 | 0.77085 | 0.47407 | 0.84945 | 0.84853 |
| LPIPS-VGG ↓ | 0.20316 | 0.34575 | 0.64648 | 0.21333 | 0.21746 |
| G1 点→网格 中位相对距离 (%) ↓ | 0.0687 | 0.1421 | 0.8392 | 0.0839 | 0.0684 |
| G1 点→网格 中位绝对距离 ↓ | 0.004019 | 0.008311 | 0.049078 | 0.004908 | 0.003997 |
| G1 均值绝对距离 ↓ | 0.010350 | 0.024002 | 0.107318 | 0.011746 | 0.010877 |
| G2 距离<1% 占比 (%) ↑ | 97.736 | 87.278 | 55.469 | 97.542 | 97.489 |
| G2 距离<0.5% 占比 (%) ↑ | 94.134 | 75.811 | 33.800 | 93.256 | 93.348 |
| G4 顶点数 | 205548 | 211274 | 120059 | 208382 | 205113 |
| G4 面数 | 370128 | 381723 | 222535 | 375966 | 370554 |
| G4 连通分量数 ↓ | 2233 | 2403 | 1192 | 2242 | 1872 |
| G4 最大分量面占比 (%) ↑ | 41.686 | 45.251 | 84.583 | 41.768 | 44.585 |
| G4 碎片分量数 (<100面) ↓ | 2137 | 2324 | 1177 | 2142 | 1787 |
| G4 碎片总面数 ↓ | 29408 | 23944 | 8343 | 26262 | 24117 |
| G4 非流形边 ↓ | 0 | 0 | 0 | 0 | 0 |
| G4 边界边 ↓ | 38596 | 35449 | 23485 | 40452 | 35934 |
| G5 绝对二面角 mean (度) ↓ | 33.6019 | 36.3355 | 34.4077 | 34.5009 | 30.2250 |
| G5 绝对二面角 P90 (度) ↓ | 75.1845 | 77.1081 | 76.5826 | 75.9158 | 72.2780 |
| 训练墙钟 (min) | 12.15 | 13.93 | 14.38 | 12.89 | 14.55 |
| 每迭代 (ms) | 91.087 | 104.463 | 107.868 | 96.686 | 109.147 |
| 峰值显存 (GB) | 7.415 | 7.625 | 8.377 | 7.585 | 7.651 |
| 末端高斯数 | 429423 | 429395 | 429416 | 429406 | 429411 |

列名对照：`base_seed0` = λ=0 基线（打了补丁但关掉 DNC）；`dnc005` / `dnc02` = 自研 DNC λ=0.05 / 0.2；
`dnc02_detach` = 自研 DNC λ=0.2 + `--dnc_detach_depth True`；`official_dnc` = 本次的官方 dn_consistency（λ=0.05）。
`summary.csv` 里的 `dnc_factor` 列对本 run 显示 `NA`，因为按主会话指定的字段名，官方组写的是
`dn_consistency_factor` 而不是 `dnc_factor`（**刻意如此**：避免在同一列里把官方正则冒充成自研 DNC）。

### 9.1 训练代价：去掉 GPU 争用后的同口径对比

`summary.csv` 里的 `train_ms_per_iter` 是整段训练（含 7001–9000 的无 DNC 段）的平均，且本 run 末尾有
GPU 争用。按训练器自己打印的 200 迭代分块耗时重算，**只取 DNC 生效段（9001–15000）并剔除被争用的分块**：

| run | 7001–9000 (ms/iter) | 9001–15000 (ms/iter, 干净块) | 相对 λ=0 基线 |
|---|---|---|---|
| `coarse_base_seed0` | 34.2 | 109.9 | — |
| `coarse_official_dnc` | 34.0 | **116.5**（剔除 2 个被争用块：292/297） | **+6.0%** |
| `coarse_dnc005` | 34.9 | 127.5 | +16.0% |
| `coarse_dnc02` | 34.6 | 132.1 | +20.3% |
| `coarse_dnc02_detach` | 35.1 | 117.1 | +6.6% |

> 官方实现更省：它把「深度 z + 法向 x,y」打包进**一次** 3 通道光栅化（`render_depth_and_normal`，
> 法向 z 分量由 `−sqrt(1−x²−y²)` 解析恢复）；自研实现另外渲染了法向图并做逐迭代 CSV / 可视化落盘。

复算命令（数字全部落盘，不是手抄）：

```bash
python scripts/eval/blocktime_from_logs.py          # 新增脚本
cat outputs/metrics/blocktime_dnc_active.json       # 其输出
```

> 口径提醒 1：`coarse_base_seed0` 跑在**上一个 PBS 作业**且 `OMP_NUM_THREADS=24`，其余四组都是 12。
> 训练循环是 GPU-bound（CPU 只做数据搬运），但严格说基线这一行的 CPU 侧条件与其余四组不同，
> 所以「+6.0% / +16.0%」应读作**量级结论**，不是精确到 0.1% 的测量。
> 口径提醒 2：只有 `coarse_official_dnc` 有被剔除的分块；其余四组 `dropped` 均为空，说明它们跑的时候独占 GPU。

---

## 10. 验收条目逐条核对

| # | 验收条目 | 期望 | 实际 | 判定 | 复算命令 |
|---|---|---|---|---|---|
| A1 | 新增文件，不改任何已有模块 | `git status` 中除他人补丁外只多出 `?? train_coarse_official_dnc.py` | 见 §2，两仓库均如此 | **PASS** | `git -C repo/SuGaR_dev status --short` |
| A2 | 官方正则代码零改动 | `coarse_density_and_dn_consistency.py` 不出现在 `git status` | 未出现，且两仓库该文件 `diff` 为空 | **PASS** | `diff repo/SuGaR/sugar_trainers/coarse_density_and_dn_consistency.py repo/SuGaR_dev/sugar_trainers/coarse_density_and_dn_consistency.py` |
| A3 | 两仓库副本一致 | md5 相同 | `ff04346a57b2f6be54be980956c79478` ×2 | **PASS** | `md5sum repo/SuGaR{,_dev}/train_coarse_official_dnc.py` |
| A4 | `py_compile` | 通过 | 两份均通过 | **PASS** | `python -m py_compile repo/SuGaR_dev/train_coarse_official_dnc.py` |
| A5 | argparse 解析 | 10 个参数值与预期逐个相等 | `ARGPARSE_TEST_OK` | **PASS** | `python <scratchpad>/test_parse_official_dnc.py` |
| A6 | seed 固定 | 日志出现 `[REPRO] ... seeded with 0` | 出现（日志第 17 行） | **PASS** | `grep REPRO logs/s34_coarse_official_dnc.log` |
| A7 | 官方 λ / start 未被改动 | 日志 `Depth-Normal consistency factor: 0.05`；iter 9001 启动 | 两者均符合 | **PASS** | `grep -n "Depth-Normal consistency factor\|Starting depth-normal consistency" logs/s34_coarse_official_dnc.log` |
| A8 | `-c` 以 `/` 结尾 | 是 | `.../gs_truck/` | **PASS** | `grep "3dgs ckpt" logs/s34_coarse_official_dnc.log` |
| A9 | `train_stats.json` 字段齐全且与自研一致 | 含 `train_wallclock_min`/`mean_time_per_iteration_ms`/`max_memory_allocated_MiB`/`n_gaussians_final`/`seed`/`regularization`/`dn_consistency_factor`/`start` | 全部存在，见 §7 | **PASS** | `cat outputs/runs/coarse_official_dnc/train_stats.json` |
| A10 | 高斯数与日志一致 | `n_gaussians_final` == 日志剪枝后数量 == render json 的 `n_gaussians` | 429411 == 429411 == 429411 | **PASS** | `grep "Pruning finished" logs/s34_coarse_official_dnc.log; python -c "import json;print(json.load(open('outputs/metrics/render_coarse_official_dnc.json'))['n_gaussians'])"` |
| A11 | mesh 参数与其余组完全一致 | `-l 0.3 -d 200000 --eval True`，Poisson depth 10，quantile 0.1 | 全部一致 | **PASS** | `cat outputs/runs/mesh_official_dnc/extract_stats.json` |
| A12 | mesh 产物可用 | `.ply` 存在且被 open3d 成功读出 V/F | V=205113 F=370554，15,278,291 B | **PASS** | `python -c "import open3d as o3d;m=o3d.io.read_triangle_mesh('outputs/runs/mesh_official_dnc/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply');print(len(m.vertices),len(m.triangles))"` |
| A13 | 评测四步全通 | S4.1–S4.4 exit=0 | 全部 exit=0 | **PASS** | `tail -3 logs/eval_coarse_official_dnc_091852.log` |
| A14 | 本 run 已进入 `summary.csv` | 出现 `official_dnc` 行 | 出现（8 行表的最后一行） | **PASS** | `grep '^official_dnc,' outputs/metrics/summary.csv` |
| A15 | 我本人未删除文件 / 未 commit / 未 pip install | 是 | 是。`repo/SuGaR_dev` HEAD 仍 `7c10c4a`；`repo/SuGaR` 的 HEAD 变成 `39c0f56`，是**主会话 09:19:16 的 commit**（作者 Find-Yit，Co-Authored-By: Claude Fable 5.1），非本执行者所为 | **PASS** | `git -C repo/SuGaR_dev log --oneline -1; git -C repo/SuGaR log -1 --format='%h %an %ad %s'` |
| A16 | 启动时独占 GPU | 启动瞬间只有本进程 | 08:58:55 启动后 `nvidia-smi` 仅本进程 | **PASS（启动时）** | 见 §11 偏离 2（训练末段与 mesh 段被他人并发） |
| A17 | 墙钟口径可复算 | 分块求和法误差 <0.1% | dnc005：13.9179 vs 13.93 min | **PASS** | `python <scratchpad>/loop_time.py logs/s34_coarse_dnc005.log` |

**总计 17 条：17 PASS / 0 FAIL**（A16 附条件，见偏离 2）。

---

## 11. 偏离与异常 [DEVIATION]

**[DEVIATION-1] GPU 排队信号的前置链条被他人两次改动，但不影响本 run 的输入**
- 08:33:01 执行者 6 的 `seq` 链在 `coarse_dnc005` 训练后中断（`scripts/run_s34_one.sh: error reading input file`），
  原因是该 shell 脚本在运行中被并发改写。执行者 6 手动补跑了 `mesh_dnc005`，执行者 7 用新脚本
  `run_s34_detach_after_mesh005.sh` 重启了 `coarse_dnc02_detach`（08:39:55 起）。
- 影响：我等待的信号推迟到 08:58:41 才出现（仍远早于 09:35 截止）。**对本 run 的输入（场景 / 3DGS 权重 /
  seed / 迭代数）零影响**，只推迟了开始时间。
- 我的应对：全程只等信号、不抢 GPU；并且**没有编辑任何正在运行的脚本**。

**[DEVIATION-2] 本 run 的训练末段与 mesh 段被他人作业并发占用 GPU0（不是我发起的）**
- 09:09:46 起，另一名执行者开始 Poisson 深度消融（`logs/s3f_*`，`mesh_base_seed0_pdauto` / `mesh_base_seed0_q0`），
  与我的训练最后约 400 迭代重叠；我的 mesh 提取（09:13:35–09:18:52）全程与其 2 个 `extract_mesh.py` 并发。
- 证据：我的训练分块耗时在最后 2 个 200-迭代块从 ~115 ms/iter 跳到 **292 / 297 ms/iter**；
  `nvidia-smi` 在 09:15 显示 3 个 `extract_mesh.py` 进程。
- 影响：**只影响耗时类指标，不影响任何精度 / 几何指标**（训练与提取都是确定性的，且我在 09:13 之前独占）。
  `train_stats.json` 里的 `train_wallclock_min=14.55` / `109.1 ms/iter` 因此**偏大**；
  §9.1 给出了剔除争用块后的同口径值 **116.5 ms/iter**。mesh 的 317 s 同样偏大，不做代价结论。
- 我没有杀掉任何他人进程，也没有重跑（重跑会再次抢卡且对精度指标无益）。

**[DEVIATION-3] `extract_mesh.py` / `sugar_extractors/coarse_mesh.py` 在我的链运行期间被他人修改**
- 09:08:32 / 09:08:45 这两个文件被改（新增 Frosting 式 `--poisson_depth auto`、`--vertices_density_quantile`
  等开关）；我的 mesh 提取在 09:13:35 启动，用的是**改后**的版本。
- 我做的核验（不是只看注释）：新增开关的默认值等价于原硬编码值，且我的 mesh 日志实打实打印
  `Poisson depth: 10` / `Vertices density quantile (cleaning quantile): 0.1` / `Only report depth: False`，
  `mesh_official_dnc/extract_stats.json` 也记录 `poisson_depth_used: 10`、`poisson_depth_is_auto: false`。
  **即提取行为与前面各组完全一致**，V/F 可与它们直接比较。
- 残留风险：我无法验证「改后代码在默认路径上与改前**逐位**等价」（没有改前的同一 `.pt` 做 A/B）。
  如需彻底排除，可用 `git stash` 前的版本对 `coarse_base_seed0/15000.pt` 重跑一次提取比对 V/F —— 但那需要占卡，
  且 `base_seed0_q0` 一行（同一 coarse 模型、只改 quantile）已间接说明默认路径没有被破坏。

**[DEVIATION-4] `mean_time_per_iteration_ms` 的分母口径**
- 主会话要求「按 8000 iter 算」，自研各组写的是 `n_iterations_done = 8001`（含第 7000 步本身）。
  本 run 按要求用 8000。两者差 0.0125%，对比表不受影响；`train_stats.json` 里用 `n_iterations_counted: 8000`
  显式标明了分母。

**[DEVIATION-5] 官方训练器不单独记录 L_dn**
- 官方实现只把 `0.05 * normal_error` 加进总损失，不打印分项，所以本 run 没有自研组那样的 `last_l_dnc`。
  只能报总损失 0.095709（λ=0 基线 0.083459，自研 λ=0.05 为 0.118488）。若 Fable 需要 L_dn 曲线，
  需要额外改官方训练器 —— 我没有改（规则：只新增、不改已有文件）。

---

## 12. 总结论（只写数字支持的，不美化）

1. **官方 dn_consistency（λ=0.05）跑通了，且是目前唯一「几何变好、渲染几乎不掉」的 DNC 组。**
   相对 λ=0 基线 `base_seed0`：PSNR 24.6201 → 24.3779（−0.24 dB）、SSIM 0.85552 → 0.84853、
   LPIPS 0.20316 → 0.21746；而 G1 中位距离 0.004019 → **0.003997**（略优）、
   连通分量 2233 → **1872（−16.2%）**、碎片分量 2137 → **1787（−16.4%）**、
   碎片总面数 29408 → **24117（−18.0%）**、G5 绝对二面角均值 33.60° → **30.22°（−10.1%）**、
   P90 75.18° → **72.28°（−3.9%）**。
2. **同样 λ=0.05，自研 DNC 明显更差**：PSNR 24.6201 → 22.4473（−2.17 dB）、SSIM → 0.77085、
   LPIPS → 0.34575；G1 中位距离 0.004019 → 0.008311（翻倍）、G2<0.5% 94.134% → 75.811%；
   碎片分量反而升到 2324。**即「深度-法向一致性」这个想法本身（2DGS）在 SuGaR coarse 阶段是有效的，
   失效的是我们的具体实现细节**，最可能的三个嫌疑（按怀疑度排序，均可证伪）：
   (a) 我们用 `1 − |cos|` 的绝对值：官方是 `1 − cos`，靠逐高斯翻向相机解决朝向歧义。绝对值会让
       「法向翻转 180°」也成为零损失解，等于放行了一族错误几何；
   (b) 我们额外渲染 3 通道法向图并在像素上再归一化，而官方只渲染法向的 x,y 再解析恢复
       `z = −sqrt(1−x²−y²)`（强制单位模长、强制朝向相机），后者对 alpha 合成后的「短法向」更稳健；
   (c) 我们的有效像素掩码（含 `‖N_raw‖>0.1`、深度梯度阈值）把物体边界大量剔除，可能反而使
       损失集中在平坦区域、鼓励整体抹平。
   **注意：以上是假设，本次实验没有做 (a)(b)(c) 的逐项消融，不能当结论写进报告。**
3. **代价**：官方版本在 DNC 生效段只比基线慢 **+6.0%**（116.5 vs 109.9 ms/iter），
   自研 λ=0.05 慢 +16.0%；峰值显存 7.651 GB vs 基线 7.415 GB（+3.2%）。
4. 本 run 的所有数字都能从 `outputs/metrics/summary.csv`、
   `outputs/metrics/{render,geometry,meshvis}_coarse_official_dnc.json`、
   `outputs/runs/coarse_official_dnc/train_stats.json` 与
   `outputs/runs/mesh_official_dnc/extract_stats.json` 复算，没有手抄的数。

## 13. 产物清单（绝对路径）

- 新增代码：`/scratch/e1351071/zju_test/repo/SuGaR_dev/train_coarse_official_dnc.py`
- 新增代码副本：`/scratch/e1351071/zju_test/repo/SuGaR/train_coarse_official_dnc.py`
- 新增脚本：`/scratch/e1351071/zju_test/scripts/run_s3e_official_dnc.sh`
- 新增脚本：`/scratch/e1351071/zju_test/scripts/eval/blocktime_from_logs.py`
  → 输出 `/scratch/e1351071/zju_test/outputs/metrics/blocktime_dnc_active.json`
- coarse 模型：`/scratch/e1351071/zju_test/outputs/runs/coarse_official_dnc/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`
- 训练统计：`/scratch/e1351071/zju_test/outputs/runs/coarse_official_dnc/train_stats.json`
- 网格：`/scratch/e1351071/zju_test/outputs/runs/mesh_official_dnc/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply`
- 提取统计：`/scratch/e1351071/zju_test/outputs/runs/mesh_official_dnc/extract_stats.json`
- 指标：`/scratch/e1351071/zju_test/outputs/metrics/render_coarse_official_dnc.{json,csv}`、
  `geometry_coarse_official_dnc.json`、`meshvis_coarse_official_dnc.json`、`summary.csv`
- 可视化：`/scratch/e1351071/zju_test/outputs/vis/coarse_official_dnc/`
- 日志：`/scratch/e1351071/zju_test/logs/s34_coarse_official_dnc.log`、`s34_mesh_official_dnc.log`、
  `s3e_official_dnc_waiter.log`、`s3e_official_dnc_chain.log`、`eval_coarse_official_dnc_091852.log`
