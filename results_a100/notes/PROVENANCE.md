# PROVENANCE.md — A100 侧 run 的来源与可比性（清单第 2 项）

> 本文回答清单第 2 项的三个问题。所有数字可追溯到本机 `logs/`、`outputs/metrics/`、`notes/`。

## 1. coarse 模型：**不是本机重训，直接复用 hopper 回传的 .pt**

本机（NSCC 4×A100）**没有重训任何模型**。全部 9 组提取 run 都加载同一个 checkpoint：

```
outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt
```

该 .pt 由 **hopper H200** 训练后回传到本机，未做任何修改（每组 run 的
`provenance_m_<TAG>.json` / `extract_stats_<TAG>.json` 里的 `coarse_model_path` 均指向它）。
本机只重跑了 **提取（extract_mesh.py）+ 剪枝（prune_coarse_model.py）+ 评测**。

### 1.1 3DGS 7k（hopper H200）
- 日志：`logs/s21_3dgs.log`；RUNLOG：`notes/RUNLOG.md` §S2.1
- 起止：`2026-09-13 05:48:58 → 05:51:11`，**墙钟 2m13s**（约 85 it/s）
- seed：`gaussian_splatting/train.py` 的 `safe_state()` 固定 `random/np/torch` 种子 = **0**
- 命令（逐字）：
```bash
cd $PROJ_ROOT/repo/SuGaR
CUDA_VISIBLE_DEVICES=0 python gaussian_splatting/train.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -m $PROJ_ROOT/outputs/baseline/gs_truck \
  --iterations 7000 --save_iterations 7000 --test_iterations 7000 --eval
```
- 关键输出：init 点数 136029；`[ITER 7000] test L1 0.03830 PSNR 23.9087`；`train L1 0.03396 PSNR 24.7644`

### 1.2 coarse SuGaR 15k（hopper H200）
- 日志：`logs/s34_coarse_base_seed0.log`；RUNLOG：`notes/RUNLOG_s34.md`
- 起止：`2026-09-13 06:15:43 → 06:28:42`，**墙钟 779 s（12.15 min）**，8001 次迭代，91.1 ms/iter，
  峰值显存 7593 MiB，最终 **429423** 个高斯
- seed：**0**（显式 `--seed 0`，日志第 17 行 `[REPRO] random / numpy / torch seeded with 0.`）
- 命令（逐字，`logs/s34_coarse_base_seed0.log` 第 14 行 `command :`）：
```bash
python train_coarse_density.py \
  -s .../data/tandt/truck -c .../outputs/baseline/gs_truck/ -i 7000 \
  -o .../outputs/runs/coarse_base_seed0 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0 --dnc_start 9000
```
  （`--dnc_factor 0` = 关闭自研 DNC，即**原版 coarse 行为**；代码为 `repo/SuGaR_dev`，
  base commit `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`）
- 数据划分：`--eval` + llffhold=8 → train 219 / test 32（共 251 张 979×546）

## 2. 关于「碎片数 2137」：**2137 是 hopper 的数字，不是本机的**

- hopper 侧原版提取（`coarse_base_seed0`）碎片 = **2137**，连通分量 2233，边界边 38596。
- **本机自己的原版提取（`base_d10_q01`，D=10/10、q=0.1，即原版默认参数）碎片 = 2094**，
  连通分量 2184，边界边 38450（`summary.csv` / `geometry_m_base_d10_q01.json`）。
- 差异 **2%**（2094 vs 2137），属**提取过程的随机性**，不是模型不同：
  - 同一模型、同一参数在本机重复两次（`base_d10_q0` vs `frosting_q0_s0`，均 D=10/q0/seed0，
    后者走 auto 路径但估出同样的 D=10）得到 1651 vs 1645 个连通分量，**差 0.4%**；
    G1/G2/G5 相对差 < 0.2%。
  - 因此本机的**噪声底约 0.5%，保守取 5%**；拓扑计数变化 > 3% 才当作信号。
  - 随机性来源：Poisson 重建前的表面采样（原版无种子；本次新增 `--extract_seed` 才可复现）、
    CUDA 光栅化原子累加非确定、以及跨机（H200 vs A100）浮点/线程差异。
- **本机所有对比一律以本机自己的 `base_d10_q01`（原版默认）与 `base_d10_q0`（q=0 基线）为基线**，
  不与 hopper 的 2137 直接相减。跨机同参差异：G1/G2/G5 < 1%，拓扑计数 2–3%（详见
  `RESULTS_m_nscc_2026-09-13.md` §3、§9）。

## 3. 硬件 / 版本 / commit

| 项 | 值 | 出处 |
|---|---|---|
| GPU | **4 × NVIDIA A100-SXM4-40GB**（driver 570.124.06，compute cap 8.0） | `ENV.md` §1 |
| CPU / 内存 | 64 核 / 503 GB | `ENV.md` §1 |
| 主机 / 作业 | `x1000c2s3b0n0`，PBS `19739513.pbs101`，队列 g3 | `ENV.md` |
| 容器 | Singularity `pytorch_23.05_py3.sif` | `ENV.md` |
| OS / 编译器 | Ubuntu 22.04.2，glibc 2.35，gcc 11.3.0，**CUDA 12.1 (V12.1.105)** | `ENV.md` §2 |
| Python | 3.10.6（venv `/scratch/users/nus/e1351071/virtualenvs/zju_test`） | `ENV.md` §3 |
| torch | **2.4.1+cu121**（torchvision 0.19.1+cu121，cudnn 90100） | `ENV.md` §3 |
| **pytorch3d** | **0.7.8** | `ENV.md` §4 |
| open3d / sklearn / numpy | 0.19.0 / 1.7.2 / 1.26.4 | `ENV.md` §4 |
| 完整冻结清单 | `results_a100/notes/pip_freeze_nscc.txt` | — |
| 改动仓库 commit | `repo/sugar-dnc-zju` → **39c0f56044bc1db5d41f2ffc34cfb8e21200f300**（2026-09-13 09:19:16，"中期快照：DNC 改动、官方 dn_consistency 包装、Frosting 自动 Poisson 深度移植…"）；本次 M1+a/M2-B 改动在该 commit 之上**未提交** | `git -C repo/sugar-dnc-zju log -1` |
| 上游 SuGaR commit | **7c10c4ae4a267dece512f5c7f40ed212a0a2ab44**（Anttwo，2024-09-24，"Minor fix."） | `git -C repo/SuGaR log -1` |

## 4. 数据
`data/tandt/truck`（3DGS 官方 tandt_db.zip，sha256 `816e62f2…d7e1`），251 张 979×546，
COLMAP sparse/0，`--eval` llffhold=8 → 219 train / 32 test。`cameras_spatial_extent = 5.847915`
（本机独立复算一致，见 `metrics/verify_independent.json`）。

## 5. 一句话可比性结论
> A100 侧与 hopper 侧共用**同一个 3DGS→coarse 链的同一份权重**（hopper H200 训练，seed 0），
> A100 只负责提取端实验；因此提取端各组之间**严格可比**（同一 .pt、同一机器、同一 seed），
> 而与 hopper 数字的绝对值对比需容忍 2–3% 的拓扑计数跨机差异。
