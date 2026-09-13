# RUNLOG — 阶段 3 / S3.4：三组正式 coarse SuGaR 对照 + mesh 提取

> 执行者 4（opus-executor）。启动 2026-09-13 06:15。
> 代码：`repo/SuGaR_dev`（已含 DNC 补丁），base commit `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`，
> 工作树 dirty = `M sugar_trainers/coarse_density.py`、`A sugar_utils/dnc_utils.py`、`M train_coarse_density.py`（即 DNC 补丁本身，未 commit）。
> **未修改任何代码，未删除任何文件，未 commit。`outputs/runs/coarse_baseline`（阶段 2 未打补丁留档）与
> `outputs/runs/mesh_baseline` 全程未被触碰**（启动前 `15000.pt` 的 md5 记录在 `logs/s34_baseline_pt_md5_before.txt`）。

## 0. 三组共同的固定量

| 项 | 值 |
|---|---|
| 场景 | `/scratch/e1351071/zju_test/data/tandt/truck` |
| 3DGS ckpt（三组共用） | `/scratch/e1351071/zju_test/outputs/baseline/gs_truck/`（**尾斜杠必需**） |
| `-i` 载入迭代 | 7000 |
| seed | 0 |
| 总迭代 | 15000（不传 `--num_iterations`，走默认） |
| `--eval` | True（llffhold=8 → 219 train / 32 test） |
| `-e/--estimation_factor` | 0.2（默认） |
| `-n/--normal_factor` | 0.2（默认） |
| `--dnc_start` | 9000（三组均显式传入） |
| mesh 提取参数 | `-l 0.3 -d 200000 --eval True`（与阶段 2 baseline 完全一致） |
| 运行脚本 | `scripts/run_s34_one.sh`（train 成功后自动串接 extract_mesh） |
| 环境 | `source /scratch/e1351071/zju_test/env.sh` 的同一 ML venv；`OMP_NUM_THREADS=24`（三组一致，继承 shell 默认） |

## 1. 三组的差异量与调度

| run | λ (`--dnc_factor`) | GPU | 训练输出目录 | mesh 输出目录 | 训练日志 | mesh 日志 |
|---|---|---|---|---|---|---|
| `coarse_base_seed0` | 0 | 0（独占） | `outputs/runs/coarse_base_seed0` | `outputs/runs/mesh_base_seed0` | `logs/s34_coarse_base_seed0.log` | `logs/s34_mesh_base_seed0.log` |
| `coarse_dnc005` | 0.05 | 1（与 dnc02 共享） | `outputs/runs/coarse_dnc005` | `outputs/runs/mesh_dnc005` | `logs/s34_coarse_dnc005.log` | `logs/s34_mesh_dnc005.log` |
| `coarse_dnc02` | 0.2 | 1（与 dnc005 共享） | `outputs/runs/coarse_dnc02` | `outputs/runs/mesh_dnc02` | `logs/s34_coarse_dnc02.log` | `logs/s34_mesh_dnc02.log` |

> mesh 目录命名沿用阶段 2 的约定（`coarse_baseline` → `mesh_baseline`），即 `coarse_X` → `mesh_X`。

PID（`logs/s34_pids.txt`）：

```
wrapper_pid python_pid run gpu
2966098 2966119 coarse_base_seed0 0
2966508 2966523 coarse_dnc005 1
2966615 2966672 coarse_dnc02 1
```

## 2. 完整命令（逐字，可直接复算）

```bash
source /scratch/e1351071/zju_test/env.sh
cd /scratch/e1351071/zju_test

# 三组同时启动（每组 train 成功后脚本内自动接 extract_mesh）
nohup env RUN=coarse_base_seed0 DNC_FACTOR=0    GPU=0 bash scripts/run_s34_one.sh > logs/s34_wrap_base_seed0.log 2>&1 &
nohup env RUN=coarse_dnc005     DNC_FACTOR=0.05 GPU=1 bash scripts/run_s34_one.sh > logs/s34_wrap_dnc005.log 2>&1 &
nohup env RUN=coarse_dnc02      DNC_FACTOR=0.2  GPU=1 bash scripts/run_s34_one.sh > logs/s34_wrap_dnc02.log 2>&1 &
```

脚本内部展开后的实际命令（三组逐字）：

```bash
cd /scratch/e1351071/zju_test/repo/SuGaR_dev

# --- coarse_base_seed0 (GPU0, λ=0) ---
python train_coarse_density.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_base_seed0 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0 --dnc_start 9000
python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_base_seed0

# --- coarse_dnc005 (GPU1, λ=0.05) ---
python train_coarse_density.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_dnc005 \
  --eval True --gpu 1 --seed 0 --dnc_factor 0.05 --dnc_start 9000
python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_dnc005/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 1 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_dnc005

# --- coarse_dnc02 (GPU1, λ=0.2) ---
python train_coarse_density.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02 \
  --eval True --gpu 1 --seed 0 --dnc_factor 0.2 --dnc_start 9000
python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 1 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_dnc02
```

## 3. 时间线（填写中）

| 事件 | 时刻 |
|---|---|
| 三组 train 启动 | `coarse_base_seed0` 06:15:43 / `coarse_dnc005` 06:15:45 / `coarse_dnc02` 06:15:47 |

（其余在运行结束后补全）

---

## 4. 单卡重跑（新作业 614430，hopper-13）

> 执行者 6 追加，2026-09-13 08:41。本节只记录 **λ=0.05 / λ=0.2 两组在新 PBS 作业上的重跑**；
> `coarse_base_seed0`（λ=0）仍是 06:15–06:35 在**上一个作业**里跑完的那一份，未重跑、未被触碰。
> 本节未修改任何代码、未删除任何文件、未 commit。

### 4.1 被杀事件（为什么要重跑）

- 上一个 PBS 作业（2×H200）在 **06:49** walltime 到期被强杀。当时 `coarse_base_seed0` 已完整跑完
  （训练 06:15:43–06:28:42 + mesh 06:28:42–06:35:38），而 `coarse_dnc005` / `coarse_dnc02` 共享 GPU1，
  跑到 14900/15000 被杀，**没有产出 15000.pt**（SuGaR 只在训练结束时存 ckpt，不存中间 ckpt）。
- 残缺产物与日志被改名保留为失败记录，**不是结果，不要引用**：
  - `outputs/runs/coarse_dnc005_killed_0649/`、`outputs/runs/coarse_dnc02_killed_0649/`
  - `outputs/runs/mesh_dnc005_killed_0649/`、`outputs/runs/mesh_dnc02_killed_0649/`（空）
  - `logs/s34_coarse_dnc005_killed_0649.log`、`logs/s34_coarse_dnc02_killed_0649.log`
- 新作业 **614430.hopper-m-02，节点 hopper-13，只有 1×H200**，故改为**顺序**重跑两组。

### 4.2 驱动脚本与命令（逐字）

```bash
# Fable 于 07:57 启动（nohup 到 logs/s34_seq_1gpu.log）
bash scripts/run_s34_seq_1gpu.sh
```
`scripts/run_s34_seq_1gpu.sh` 内容：先 `RUN=coarse_dnc02 DNC_FACTOR=0.2 GPU=0 bash scripts/run_s34_one.sh`，
再 `RUN=coarse_dnc005 DNC_FACTOR=0.05 GPU=0 bash scripts/run_s34_one.sh`。展开后的实际命令：

```bash
cd /scratch/e1351071/zju_test/repo/SuGaR_dev

# --- coarse_dnc02 (GPU0, λ=0.2) ---
python train_coarse_density.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0.2 --dnc_start 9000
python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_dnc02

# --- coarse_dnc005 (GPU0, λ=0.05) ---
python train_coarse_density.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
  -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_dnc005 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0.05 --dnc_start 9000
# ↓ 这一步 seq 链没跑成（见 4.4），由执行者 6 手动补跑，命令逐字如下：
python extract_mesh.py \
  -s /scratch/e1351071/zju_test/data/tandt/truck \
  -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ -i 7000 \
  -m /scratch/e1351071/zju_test/outputs/runs/coarse_dnc005/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 --eval True --gpu 0 \
  -o /scratch/e1351071/zju_test/outputs/runs/mesh_dnc005
```

评测（两组，GPU0）：
```bash
source /scratch/e1351071/zju_test/env.sh
bash scripts/eval/run_eval_for_run.sh coarse_dnc02 \
  /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  /scratch/e1351071/zju_test/outputs/runs/mesh_dnc02/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply 0
bash scripts/eval/run_eval_for_run.sh coarse_dnc005 \
  /scratch/e1351071/zju_test/outputs/runs/coarse_dnc005/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  /scratch/e1351071/zju_test/outputs/runs/mesh_dnc005/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply 0
python scripts/eval/summarize.py
```

### 4.3 时间线（全部来自日志里的 date 打点）

| 事件 | 起 | 止 | 墙钟 | 退出码 | 日志 |
|---|---|---|---|---|---|
| SEQ 启动（hopper-13, job 614430.hopper-m-02） | 07:57:45 | — | — | — | `logs/s34_seq_1gpu.log` |
| `coarse_dnc02` 训练 | 07:57:45 | 08:12:56 | **911 s（15 m 11 s）** | 0 | `logs/s34_coarse_dnc02.log` |
| `coarse_dnc02` mesh 提取 | 08:12:56 | 08:18:30 | **334 s** | 0 | `logs/s34_mesh_dnc02.log` |
| dnc02 chain 结束 | — | 08:18:30 | — | 0 | `logs/s34_seq_1gpu.log` |
| `coarse_dnc02` 评测（S4.1/4.2/4.3/4.4） | 08:18:52 | 08:19:43 | 51 s | 0 | `logs/eval_coarse_dnc02_081852.log` |
| `coarse_dnc005` 训练 | 08:18:31 | 08:33:00 | **835.8 s（13 m 56 s，train_stats 计）** | 0 | `logs/s34_coarse_dnc005.log` |
| dnc005 chain 异常结束（mesh 未执行） | — | 08:33:01 | — | **2** | `logs/s34_seq_1gpu.log` |
| `coarse_dnc005` mesh 手动补跑 | 08:34:08 | 08:39:54 | **346 s** | 0 | `logs/s34_mesh_dnc005.log` |
| `coarse_dnc005` 评测 | 08:40:23 | 08:41:19 | 56 s | 0 | `logs/eval_coarse_dnc005_084023.log` |
| （对照）`coarse_base_seed0` 训练/mesh，上一个作业 | 06:15:43 / 06:28:42 | 06:28:42 / 06:35:38 | 779 s / 416 s | 0 / 0 | `logs/s34_coarse_base_seed0.log`、`logs/s34_mesh_base_seed0.log` |

### 4.4 事故：dnc005 的 mesh 步骤被跳过（chain exit=2）

- **现象**：`logs/s34_seq_1gpu.log` 记录 `=== dnc005 chain exit=2 ===`；`logs/s34_mesh_dnc005.log`
  当时**根本不存在**（不是空文件），说明 `run_s34_one.sh` 的 mesh 段一行都没执行到。
  但训练本身完全正常：日志有 `Final model saved.` 与 `[STATS] 8001 iterations in 13.93 min ...`，
  `15000.pt`（304,019,606 B）与 `train_stats.json` 都已落盘。
- **根因**（由主会话定位并告知）：`scripts/run_s34_one.sh` 的 mtime = **08:18:50**，即 bash 正在
  逐段读取该脚本时，另一位执行者在编辑同一个文件，bash 报 "error reading input file" 并以 2 退出，
  于是 TRAIN END 的 echo 与整个 MESH 段都没有执行。**与训练/显存/数据无关。**
- **处置**：执行者 6 按 4.2 的逐字命令手动补跑 mesh，参数与 `coarse_dnc02` / `coarse_base_seed0`
  **完全一致**（`-l 0.3 -d 200000 --eval True`，同一 3DGS ckpt、同一 `-i 7000`），08:39:54 成功产出
  `outputs/runs/mesh_dnc005/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply`（15,737,699 B）。

### 4.5 本次重跑的并发与环境偏差（影响"训练代价"列的可比性，报告需注明）

| 编号 | 偏差 | 证据 | 影响 |
|---|---|---|---|
| DEV-1 | `OMP_NUM_THREADS` 不一致：`coarse_base_seed0`=**24**，`coarse_dnc005`/`coarse_dnc02`=**12** | 三个训练日志头部第 13 行 | 只影响 CPU 侧耗时，**ms/iter 与墙钟的组间比较不是严格受控实验**；不影响权重与指标 |
| DEV-2 | 运行节点不同：`coarse_base_seed0` 在上一个作业的节点（2×H200，同时有 3 个训练进程共用 24 核）；两组 λ 在 hopper-13（1×H200，单进程） | `logs/s34_seq_1gpu.log` 记 host=hopper-13；base_seed0 日志无 host 记录 | 同上，墙钟/ms per iter 只能作**量级参考** |
| DEV-3 | `repo/SuGaR_dev` 的代码在两组 λ 之间被改动：`train_coarse_density.py` 与 `sugar_trainers/coarse_density.py` mtime = **08:17:14**（在 dnc02 07:57 启动之后、dnc005 08:18 启动之前），新增 `--dnc_detach_depth`（默认 False） | `logs/s34_coarse_dnc005.log:32` 有 `[DNC] Detach depth for N_d: False (...original DNC behaviour)`，dnc02 日志无此行；`train_stats.json` 里 dnc005 多出 `"dnc_detach_depth": false` 字段 | 该改动在默认值下是**语义空操作**（`dnc_depth_for_loss = dnc_depth.detach() if dnc_detach_depth else dnc_depth`，False 分支与原实现逐字等价），但**三组并非字节一致的工作树**，属实报告 |
| DEV-4 | `coarse_dnc02` 的评测（08:18:52–08:19:43）与 `coarse_dnc005` 训练前 72 s 重叠（同一张 GPU0） | 两者时间戳 | dnc005 训练墙钟/ms per iter 可能被拉高约 1 分钟量级；评测数值本身与并发无关（确定性已由执行者 3 验证） |
| DEV-5 | `coarse_dnc005` 的 mesh 补跑（08:34:08–08:39:54）与执行者 7 的 `coarse_dnc02_detach` 训练共享 GPU0 | `nvidia-smi` 在 08:33 记录 pid 1336525 = `train_coarse_density.py ... coarse_dnc02_detach --dnc_detach_depth True` | 只影响 mesh 提取耗时（346 s vs dnc02 的 334 s），几何结果与并发无关 |
| DEV-6 | `coarse_baseline`（阶段 2 未打补丁留档）没有 `train_stats.json` | 执行者 3 已记录（`SELF_CHECK_阶段4_脚本.md` N3） | summary.csv 该行训练代价列为 NA |

### 4.6 本节产出物路径

```
outputs/runs/coarse_dnc02/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt   304,034,390 B
outputs/runs/coarse_dnc02/train_stats.json
outputs/runs/coarse_dnc02/dnc_log.csv                       (60 行 + 表头，iter 9100..15000)
outputs/runs/coarse_dnc02/dnc_vis/                          (30 个 PNG，iter 10000..15000)
outputs/runs/mesh_dnc02/sugarmesh_..._level03_decim200000.ply  9,016,290 B
outputs/runs/coarse_dnc005/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt 304,019,606 B
outputs/runs/coarse_dnc005/train_stats.json
outputs/runs/coarse_dnc005/dnc_log.csv                      (60 行 + 表头，iter 9100..15000)
outputs/runs/coarse_dnc005/dnc_vis/                         (30 个 PNG，iter 10000..15000)
outputs/runs/mesh_dnc005/sugarmesh_..._level03_decim200000.ply 15,737,699 B
outputs/metrics/{render,geometry,meshvis}_coarse_dnc02.json + render_coarse_dnc02.csv
outputs/metrics/{render,geometry,meshvis}_coarse_dnc005.json + render_coarse_dnc005.csv
outputs/metrics/summary.csv                                 (5 行 × 41 列)
outputs/vis/coarse_dnc02/  outputs/vis/coarse_dnc005/       (各 20 个 PNG)
```

*本节最后更新：2026-09-13 08:41，执行者 6。*
