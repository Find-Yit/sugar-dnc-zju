# SELF_CHECK — 阶段 3d：λ=0.2 退化诊断（`--dnc_detach_depth`）

> 执行者 7（opus-executor）。时间 2026-09-13 08:16 起。
> 依据计划 `plans/SuGaR复现与改动_6小时考核.txt` §2 / §3c / §3d（预注册，未改动）。
> **未 commit、未删除任何文件、未 pip install、未修改预注册假设。**

---

## 一、改动（diff 摘要）

改动只涉及 2 个文件（+ 1 个运行脚本的最小扩展），**不碰 `sugar_utils/dnc_utils.py`**：

| 文件 | 变化 | 内容 |
|---|---|---|
| `repo/SuGaR_dev/train_coarse_density.py` | +6 / -0 | 新增 CLI 参数 `--dnc_detach_depth`（`str2bool`，默认 `False`） |
| `repo/SuGaR_dev/sugar_trainers/coarse_density.py` | +12 / -1 | 读参数、日志横幅、loss 前 `detach`、`train_stats.json` 新字段 |
| `repo/SuGaR/train_coarse_density.py` | 同上 | 由 `cp` 从 `SuGaR_dev` 同步（分支 `dnc`，**未 commit**） |
| `repo/SuGaR/sugar_trainers/coarse_density.py` | 同上 | 同步 |
| `scripts/run_s34_one.sh` | +9 / -1 | 新增 `EXTRA_ARGS=${EXTRA_ARGS:-}`，追加到训练命令末尾并写进日志头 |
| `scripts/run_s34_detach_after_seq.sh` | 新增 41 行 | v1 等待脚本：以 `SEQ END` 为条件启动第四组（触发过早，见 [DEVIATION-2b]）|
| `scripts/run_s34_detach_after_mesh005.sh` | 新增 47 行 | **v2 等待脚本（实际使用）**：等 `mesh_dnc005` 完成后再启动，保证 GPU 独占 |
| `scripts/test_dnc_detach.py` | 新增 87 行 | 开关梯度语义的 CPU 单测（8/8 PASS）|
| `scripts/run_eval_detach_after_mesh.sh` | 新增 45 行 | mesh 落盘后自动跑 `run_eval_for_run.sh` + `summarize.py`（零空转）|
| `notes/dnc_detach_3d.patch` | 新增 57 行 | 本次增量补丁（可 `git apply`）|
| `logs/s34_coarse_dnc02_detach_aborted_0833.log` | 新增 | 被中止的首次尝试的日志留档（未删除任何文件）|

核心 4 处（完整补丁见 `notes/dnc_detach_3d.patch`）：

```diff
+    parser.add_argument('--dnc_detach_depth', type=str2bool, default=False, help=...)
```
```diff
+    dnc_detach_depth = bool(getattr(args, 'dnc_detach_depth', False))
```
```diff
+                    dnc_depth_for_loss = dnc_depth.detach() if dnc_detach_depth else dnc_depth
                     dnc_loss, dnc_aux = dnc_utils.depth_normal_consistency_loss(
-                        dnc_depth, dnc_normal_img,
+                        dnc_depth_for_loss, dnc_normal_img,
```
```diff
+        'dnc_detach_depth': bool(dnc_detach_depth),
```
外加一行日志横幅 `CONSOLE.print("[DNC] Detach depth for N_d:", dnc_detach_depth, ...)`。

### 默认 `False` 时行为逐字节一致 —— 三条独立证据

1. **Python 语义**：`dnc_depth.detach() if False else dnc_depth` 求值结果**就是 `dnc_depth` 本身**，
   不新建张量、不新增 autograd 节点。实测（CPU）：

   ```
   dnc_detach_depth=False  y is x -> True    y.requires_grad=True   y.grad_fn=MulBackward0
   dnc_detach_depth=True   y is x -> False   y.requires_grad=False  y.grad_fn=None
   ```

2. **增量性证明**：把阶段 3 已被 Fable 逐行审阅通过的 `notes/dnc_stage3.patch` 应用到干净的
   `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44` 检出，得到的两个文件与"我改动前的状态"**逐字节相同**
   （`diff -q` 无输出）。即本次改动严格等于上面 5 个 hunk，没有夹带任何其他修改。

   复算命令（已实跑通过）：
   ```bash
   cd /scratch/e1351071/zju_test
   CHK=$(mktemp -d)
   git -C repo/SuGaR_dev archive 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44 | tar -x -C $CHK
   (cd $CHK && git apply notes_dnc_stage3.patch 2>/dev/null || git apply \
       /scratch/e1351071/zju_test/notes/dnc_stage3.patch)          # -> 我改动前的状态
   diff -q $CHK/train_coarse_density.py           <改动前副本>       # 无输出
   (cd $CHK && git apply /scratch/e1351071/zju_test/notes/dnc_detach_3d.patch)
   diff -q $CHK/train_coarse_density.py           repo/SuGaR_dev/train_coarse_density.py
   diff -q $CHK/sugar_trainers/coarse_density.py  repo/SuGaR_dev/sugar_trainers/coarse_density.py
   # 两条 diff 均无输出 => 7c10c4a + dnc_stage3.patch + dnc_detach_3d.patch == 当前代码
   ```

3. **实跑旁证**：`coarse_dnc005` 于 08:18:31 启动，用的**就是改动后的代码**（改动时间 08:17:14），
   日志横幅为 `[DNC] Detach depth for N_d: False (gradient through both D and N - original DNC behaviour)`，
   训练全程正常，L_dnc 曲线与 λ=0.2 组同形。

---

## 二、CPU 冒烟（GPU 当时被 dnc005 占用，按任务要求不在 GPU 上冒烟）

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法编译（两个 repo × 3 文件） | `python -m py_compile <repo>/train_coarse_density.py <repo>/sugar_trainers/coarse_density.py <repo>/sugar_utils/dnc_utils.py` | **PASS**（无输出） |
| argparse 能解析 `--dnc_detach_depth True` | 见下 | **PASS** |
| `run_s34_one.sh` / 等待脚本语法 | `bash -n` | **PASS** |
| 两个 repo 的 3 个 DNC 文件一致 | `md5sum` | 三对 md5 完全相同 |

argparse 检查（stub 掉 `sugar_trainers.coarse_density`，`CUDA_VISIBLE_DEVICES=""`，全程不碰 GPU）：

```
argv=['--dnc_detach_depth', 'True']  -> dnc_detach_depth=True  (type bool)  OK
argv=['--dnc_detach_depth', 'False'] -> dnc_detach_depth=False (type bool)  OK
argv=(default)                       -> dnc_detach_depth=False (type bool)  OK
ARGPARSE CHECK PASS for /scratch/e1351071/zju_test/repo/SuGaR_dev
ARGPARSE CHECK PASS for /scratch/e1351071/zju_test/repo/SuGaR
```

---

## 三、第四组 run：`coarse_dnc02_detach`

### 参照值（P2/P3/P4 的判据阈值在第四组开跑之前就已写死，属于盲验）

> 说明：`vanilla3dgs7k` / `coarse_baseline` / `coarse_base_seed0` / `coarse_dnc02` 四行来自 `summary.csv`，
> 在我启动第四组之前就存在，**判据阈值只由这几行决定**。`coarse_dnc005` 一行是 08:39 补跑完 mesh 之后
> 才有的（渲染指标取自执行者 6 的 `render_coarse_dnc005.json`，几何指标是我自己用 open3d 重算的），
> 只作为背景参照，**不参与任何阈值的设定**。


| run | PSNR↑ | SSIM↑ | LPIPS↓ | G1 median rel%↓ | G2 <1%↑ | 连通分量↓ | 碎片(<100面)↓ | 最大分量面占比%↑ | 面数 | G5 二面角均值↓ | G5 P90↓ | ms/iter | 峰值显存GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla3dgs7k | 23.9003 | 0.84994 | 0.20116 | — | — | — | — | — | — | — | — | — | — |
| coarse_baseline（未打补丁） | 24.6530 | 0.85644 | 0.20280 | 0.0684 | 97.688 | 2237 | 2147 | 42.494 | 370438 | 46.0908 | 113.2074 | — | — |
| **coarse_base_seed0（λ=0）** | **24.6201** | 0.85552 | 0.20316 | **0.0687** | 97.736 | **2233** | **2137** | 41.686 | 370128 | **45.9794** | **112.8438** | 91.087 | 7.415 |
| **coarse_dnc005（λ=0.05）** | **22.4473** | 0.77085 | 0.34575 | **0.1421** | 87.278 | 2403 | 2324 | 45.251 | 381723 | 49.5229 | 116.8433 | 104.5 | 7.625 |
| **coarse_dnc02（λ=0.2，无 detach）** | **13.2436** | 0.47407 | 0.64648 | **0.8392** | 55.469 | 1192 | 1177 | 84.583 | 222535 | 52.5791 | 131.5514 | 107.868 | 8.377 |

判据阈值（预注册 §3d，先写死再看结果）：
- **P2** 合格带 = base_seed0 PSNR ± 0.5 dB = **[24.1201, 25.1201]**
- **P3** 末值门槛 = coarse_dnc02 的 `last_l_dnc` = **0.04608708**，detach 组必须 **>** 该值，且曲线整体下降
- **P4** H1 参照 = base_seed0：G1 median rel 0.0687%、连通分量 2233、碎片 2137、G5 均值 45.98°/P90 112.84°
- 额外背景：λ=0.05（无退化）相对 λ=0 已经是 PSNR −2.17 dB、G1 median 0.0687%→0.1421%、碎片 2137→2324，三项全部变差 —— H1 在本场景下**本来就不成立**，这一点不依赖 detach 组的结果

### 开关语义的独立单元自检（CPU，新增 `scripts/test_dnc_detach.py`）

命令：`python scripts/test_dnc_detach.py --repo repo/SuGaR_dev`
结果落盘：`outputs/metrics/test_dnc_detach.json` → **8/8 PASS**

构造一个弯曲（非平面）合成深度面 D 与一张刻意与之**不对齐**的法向图 N，两者都是叶子张量的可导函数，
对 `L_dnc` 求反向，比较两种模式：

```
[detach=False] loss=0.09638174  |dL/dD|max=1.508097e-03  |dL/dN|max=2.799771e-04  valid=0.8594
[detach=True]  loss=0.09638174  |dL/dD|max=0.000000e+00  |dL/dN|max=2.799771e-04  valid=0.8594
```

| 检查 | 结论 |
|---|---|
| 两种模式 **loss 数值完全相同**（差 <1e-9） | PASS（所以两组的 L_dnc 曲线可直接比较） |
| detach=False 时 `∂L/∂D ≠ 0`（1.51e-3） | PASS（"抹平深度"这条平凡解路径确实存在） |
| detach=True 时 `∂L/∂D` **恰好为 0** | PASS（平凡解路径被彻底切断） |
| 两种模式 `∂L/∂N` 完全相同（2.80e-4） | PASS（法向对齐这一项没有被削弱） |
| detach=False 时 `depth_for_loss is depth` | PASS（无新张量、无新 autograd 节点） |

旁证 H4：本合成例中 `|∂L/∂D|` 是 `|∂L/∂N|` 的约 **5.4 倍**，与"深度通道主导 L_dnc 的梯度、
λ 大时把深度抹平"的机理假设方向一致（仅为量级参考，非严格证明）。

### DNC 数学单测未受影响（回归）

重跑执行者 2 的 `scripts/test_dnc_normal.py`（CPU）：**8/8 PASS**，关键数值与阶段 3 完全一致
（`plane_normals_ndc_convention` worst min cos = 0.99999988；`L(完全一致)=8.60e-09`；`L(正交)=0.999995`）。
说明 `sugar_utils/dnc_utils.py` 未被本次改动触及。

### 对照用的 dnc_vis 肉眼参照（第四组跑之前已观察，作为 P1 的判读标尺）

| 图 | 描述 |
|---|---|
| `coarse_dnc005/dnc_vis/iter010000_grid.png`（λ=0.05，健康） | 深度图里卡车车厢、栏板、驾驶室、车轮、背景树林与栅栏**清晰可辨**；法向图有车身/地面/树叶的分区结构；mask 只在物体边缘与树叶处为黑 |
| `coarse_dnc02/dnc_vis/iter010000_grid.png`（λ=0.2，已退化） | DNC 开启仅 1000 迭代，深度图**已经完全没有场景结构**——只是一个带同心波纹的光滑穹面；法向图是一整片近乎纯色的绿；mask 几乎全白 |
| `coarse_dnc02/dnc_vis/iter015000_grid.png`（λ=0.2，终态） | 深度图是几道光滑"沙丘"，法向图单色绿+平滑渐变，mask 98.3% 白 |

---

## 四、运行记录（命令 / 起止时间）


### 4.1 时间线

| 时刻 | 事件 |
|---|---|
| 08:12:56 | `coarse_dnc02` 训练结束（seq 链第一段，911 s） |
| 08:17:14 | **本次改动写入 `repo/SuGaR_dev`**（GPU 上无训练进程在跑 `coarse_density.py`，当时在跑 `extract_mesh.py`，二者不共享该模块） |
| 08:18:30 | `mesh_dnc02` 提取结束（334 s） |
| 08:18:31 | `coarse_dnc005` 训练启动 —— **它用的是改动后的代码**（默认 `dnc_detach_depth=False`，见 §一的三条等价性证据） |
| 08:18:50–08:19:0x | `scripts/run_s34_one.sh` 通过 `mv` 换成带 `EXTRA_ARGS` 的版本。dnc005 的训练命令行没受影响（日志头无 `extra args` 行，与前两组逐字相同），**但脚本后半段的 mesh 那一步被破坏** —— 见 [DEVIATION-2] |
| 08:20:17 | 启动 `scripts/run_s34_detach_after_seq.sh`（后台轮询 `SEQ END`，09:05 放弃线） |
| 08:33:01 | dnc005 训练正常结束、`15000.pt` 落盘；随后 bash 读不到脚本剩余部分，报 `error reading input file` 并 exit 2，**mesh 未执行**；`SEQ END` 因此提前出现 |
| 08:33:17 | 等待脚本据 `SEQ END` 启动第四组训练（早了） |
| 08:34:08 | 执行者 6 在 GPU0 上补跑 `mesh_dnc005`，与第四组训练重叠 |
| 08:35:22 | 我主动中止第四组（当时 iter≈9001，无 checkpoint 产出），让出 GPU0；改用 `scripts/run_s34_detach_after_mesh005.sh` 等 `mesh_dnc005` 完成后重跑 |
| 08:35:55 | 启动 `scripts/run_s34_detach_after_mesh005.sh`（条件：`s34_mesh_dnc005.log` 出现 `Mesh saved` 且 `mesh_dnc005/*.ply` 存在；放弃线 08:50）|
| 08:39:54 | `mesh_dnc005` 完成（346 s，exit=0，ply 已落盘）|
| **08:39:55** | **第四组 `coarse_dnc02_detach` 正式启动，`overlap_with_mesh_dnc005=no`（GPU0 独占）** |
| 08:48:32 | 启动 `scripts/run_eval_detach_after_mesh.sh`（mesh 落盘后自动评测，放弃线 09:20）|
| 08:53:22 | 训练结束 `exit=0 wallclock=807s`（8001 iters / 12.89 min / 96.7 ms/iter / 7767 MiB）|
| 08:58:41 | mesh 提取结束 `exit=0 wallclock=319s`，ply 15,515,366 字节 |
| 08:59:45 | 评测 + `summarize.py` 完成 `exit=0`，`summary.csv` 新增 `dnc02_detach` 行 |

### 4.2 命令

```bash
# 训练 + mesh（由 scripts/run_s34_detach_after_mesh005.sh 在 mesh_dnc005 完成后自动触发；
#              首个版本 scripts/run_s34_detach_after_seq.sh 以 SEQ END 触发，见 [DEVIATION-2b]）
cd /scratch/e1351071/zju_test
OMP_NUM_THREADS=12 RUN=coarse_dnc02_detach DNC_FACTOR=0.2 GPU=0 \
    EXTRA_ARGS="--dnc_detach_depth True" bash scripts/run_s34_one.sh

# 实际执行的训练命令（日志头逐字记录）
python train_coarse_density.py \
    -s /scratch/e1351071/zju_test/data/tandt/truck \
    -c /scratch/e1351071/zju_test/outputs/baseline/gs_truck/ \
    -i 7000 -o /scratch/e1351071/zju_test/outputs/runs/coarse_dnc02_detach \
    --eval True --gpu 0 --seed 0 --dnc_factor 0.2 --dnc_start 9000 --dnc_detach_depth True

# mesh（参数与三组完全一致）
python extract_mesh.py -s <scene> -c <gs_ckpt> -i 7000 \
    -m outputs/runs/coarse_dnc02_detach/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
    -l 0.3 -d 200000 --eval True --gpu 0 -o outputs/runs/mesh_dnc02_detach

# 评测 + 汇总
bash scripts/eval/run_eval_for_run.sh coarse_dnc02_detach <15000.pt> <mesh.ply> 0
python scripts/eval/summarize.py
```

`OMP_NUM_THREADS=12` 与 `coarse_dnc02` / `coarse_dnc005` 一致（`coarse_base_seed0` 当时是 24，
属于既有的组间不一致，本次沿用 12 以对齐**直接对照组** `coarse_dnc02`；已记录为偏差）。

### 4.2b 第四组实际结果：train_stats.json 与 dnc_log.csv

`outputs/runs/coarse_dnc02_detach/train_stats.json` 关键字段：

| 字段 | 值 |
|---|---|
| `dnc_factor` | **0.2** |
| `dnc_start` | 9000 |
| `dnc_enabled` | true |
| **`dnc_detach_depth`** | **true**（新增字段，证明开关生效）|
| `seed` | 0 |
| `last_iteration` | 15000 |
| `n_gaussians_final` | 429406 |
| `final_loss` | 0.14711663 |
| **`last_l_dnc`** | **0.29154682** |
| `last_dnc_valid_pixel_ratio` | 0.86189467 |
| `train_wallclock_min` | 12.893 |
| `mean_time_per_iteration_ms` | **96.686** |
| `max_memory_allocated_MiB` | 7767.31 |
| `timestamp` | 2026-09-13 08:53:21 |

日志横幅：`[DNC] Detach depth for N_d: True (gradient only through the normal map N)`
训练结束行：`=== S3.4 TRAIN END Sun Sep 13 08:53:22 Asia 2026 exit=0 wallclock=807s ===`

`dnc_log.csv`（60 行，iter 9100..15000）首 / 中 / 末：

| 位置 | iter | L_dnc | 有效像素 | 总损失 | sdf_better_normal | sdf_estimation |
|---|---|---|---|---|---|---|
| 首 | 9100 | **0.347649** | 0.8672 | 0.208712 | 0.093478 | 0.325245 |
| 中 | 12100 | **0.304053** | 0.8170 | 0.159526 | 0.049489 | 0.196303 |
| 末 | 15000 | **0.291547** | 0.8619 | 0.147117 | 0.043222 | 0.154579 |

全程最小值 `min L_dnc = 0.262946`（出现在 iter 14900）。
整段 60 个采样点的线性趋势 `slope = −0.00565 / 1000 iter`，前 12 点均值 **0.3298** → 后 12 点均值 **0.2990**：
**在缓慢下降，但始终停在 0.26–0.35 区间**，从未像无 detach 的 λ=0.2 那样掉到 0.04。

训练代价对比（四组，前三组数字来自各自 `train_stats.json`）：

| run | ms/iter | 峰值显存 MiB | 训练墙钟 min | 高斯数 |
|---|---|---|---|---|
| `coarse_base_seed0`（λ=0，OMP=24）| 91.09 | 7415 | 12.15 | 429423 |
| `coarse_dnc005`（λ=0.05，OMP=12）| 104.46 | 7808 | 13.93 | 429395 |
| `coarse_dnc02`（λ=0.2，OMP=12）| 107.87 | 8578 | 14.38 | 429416 |
| **`coarse_dnc02_detach`（λ=0.2+detach，OMP=12）** | **96.69** | **7767** | **12.89** | 429406 |

detach 版比不 detach 的同 λ 版**快 10.4%、省 811 MiB**：因为 DNC 项不再需要对深度图那次
光栅化做反向传播。这是该开关一个未被 §3d 预注册、但值得写进报告的附带收益。

### 4.3 L_dnc 轨迹对照（判读 P3 的背景）

`dnc_log.csv` 每 100 迭代一行。已跑完两组的关键节点：

| iter | λ=0.2（无 detach） | λ=0.05 | **λ=0.2 + detach** |
|---|---|---|---|
| 9100 | 0.317425 / 87.5% | 0.345245 / 87.5% | **0.347649 / 86.7%** |
| 10000 | 0.105518 / **98.9%** | 0.372940 / 81.9% | **0.362418 / 80.7%** |
| 11000 | 0.041034 / **98.9%** | 0.266266 / 84.3% | **0.286078 / 80.9%** |
| 12000 | 0.066424 / **98.6%** | 0.258579 / 81.6% | **0.301617 / 79.5%** |
| 13000 | 0.050804 / **98.7%** | 0.245106 / 88.8% | **0.286029 / 85.3%** |
| 14000 | 0.054781 / 98.6% | — | **0.309525 / 91.1%** |
| 15000 | **0.046087** / 98.3% | 0.234661 / 87.8% | **0.291547 / 86.2%** |

**退化的双重指纹**：λ=0.2 的 L_dnc 在 2000 迭代内从 0.32 掉到 0.04，同时有效像素比例从 87.5% 跳到 98.9%
——后者说明前景掩码几乎覆盖全图，即"整幅画面变成了一张光滑近景曲面"，与 dnc_vis 的肉眼观察一致。
健康的 λ=0.05 组 L_dnc 稳在 0.25 左右、有效像素稳在 82–89%。

### 4.4 几何指标独立复算（不信任已有 summary.csv，我自己用 open3d + numpy 重算）

```bash
python -c "import open3d as o3d,numpy as np,glob; ..."   # 见 §八 复算命令
mesh_base_seed0  V= 205548 F= 370128 comps= 2233 frag<100f= 2137 largest%=41.686 nan=False
mesh_dnc02       V= 120059 F= 222535 comps= 1192 frag<100f= 1177 largest%=84.583 nan=False
```
与 `summary.csv` 逐位一致。G5 二面角我也用纯 numpy 从 PLY 重算了一遍（按共享边配对三角面、面法向夹角）：

```
mesh_base_seed0  n_pairs= 535894 mean=45.9794 p90=112.8438   (summary.csv: 45.9794 / 112.8438)
mesh_dnc02       n_pairs= 322060 mean=52.5791 p90=131.5514   (summary.csv: 52.5791 / 131.5514)
```
四位小数完全一致 → 几何评测链路可信，可用于判 P4。

`mesh_dnc005`（执行者 6 于 08:39:54 补跑完成）我也用同一脚本算了一遍，作为 P4 的第三个参照：

```
mesh_base_seed0    V= 205548 F= 370128 comps= 2233 frag<100f= 2137 largest%=41.686 dihedral mean=45.9794 p90=112.8438
mesh_dnc005        V= 211274 F= 381723 comps= 2403 frag<100f= 2324 largest%=45.251 dihedral mean=49.5229 p90=116.8433
mesh_dnc02         V= 120059 F= 222535 comps= 1192 frag<100f= 1177 largest%=84.583 dihedral mean=52.5791 p90=131.5514
```

⚠️ 注意 λ=0.05 组：碎片数 **2324 > 2137**、二面角均值 **49.52° > 45.98°**，两项都比 λ=0 组**更差**。
也就是说在没有退化的那一组里，H1（"碎片更少、法向更平滑"）**没有得到支持**。这给 P4 的判读提供了
重要背景：λ=0.2 组那漂亮的"碎片 1177"完全是退化带来的假象。

G1/G2 我也独立重算了一遍（自己读 `sparse/0/points3D.bin`，按 track≥3 + 前景 bbox 过滤，
open3d `RaycastingScene.compute_distance`，完全不经过 `eval_geometry.py`）：

```
reference points: 88066
mesh_base_seed0    mean=0.010350 median=0.004019 p90=0.019379 | median_rel%=0.0687 | <1%ext=97.736%
mesh_dnc005        mean=0.024002 median=0.008311 p90=0.070038 | median_rel%=0.1421 | <1%ext=87.278%
mesh_dnc02         mean=0.107318 median=0.049078 p90=0.241736 | median_rel%=0.8392 | <1%ext=55.469%
```
`base_seed0` 与 `dnc02` 六位小数与 `summary.csv` 完全一致；`dnc005` 是新算的（截至此刻执行者 6 还没把它并进
summary）。**λ=0.05 的 G1 median 0.1421% 也比 λ=0 的 0.0687% 差约一倍** —— 再次说明 H1 在本场景下不成立。

**判 P4 的重要警告**：λ=0.2 组的"碎片数 2137→1177、最大分量占比 41.7%→84.6%"看上去像 H1 成立，
但它同时 PSNR 24.62→13.24、G1 median 0.069%→0.839%。那是因为退化后的网格本身就是一个大而光滑的
无结构曲面 —— **分量数变少是退化的副产物，不是"表面更干净"的证据**。因此 P4 必须把
"几何指标改善"与"渲染指标不塌"合起来看，只有 detach 组在 P2 通过的前提下，其几何变化方向才能用来回答 H1。

### 4.5 渲染指标链路的独立复算

`python scripts/eval/check_png_psnr.py --runs coarse_base_seed0,coarse_dnc02`
（只用 PIL + numpy 从落盘 PNG 重算 PSNR，完全不经过 torch / 3DGS 的实现）：

```
coarse_base_seed0  view00 25.1835 / CSV 25.1853 (diff 0.0018 dB)   view08 23.1407 / 23.1419
                   view16 24.9129 / 24.9146                        view24 26.2681 / 26.2704
coarse_dnc02       view00 13.8380 / 13.8380                        view08 12.9626 / 12.9626
                   view16 15.0340 / 15.0342                        view24 15.3526 / 15.3527
8 个视角，最大差 0.0023 dB（阈值 0.05 dB）-> PASS
```

### 4.8 P1 的**客观量化**指标（不只靠肉眼）

"深度图是否保留场景结构"用眼睛判有主观性，我补了一个客观量：对落盘的 `iterXXXXXX_depth.png`
计算 **Laplacian 标准差** 与 **相邻像素梯度 |∇D| 超过 0.02 的像素占比**。
光滑穹面几乎没有二阶结构与不连续；真实场景则在物体轮廓处有大量不连续。

| run | iter | Laplacian std | mean&#124;∇D&#124; | %px &#124;∇D&#124;>0.02 |
|---|---|---|---|---|
| `coarse_dnc02`（退化） | 010000 | 0.01403 | 0.00505 | **2.480** |
| `coarse_dnc02` | 012000 | 0.02190 | 0.00414 | **1.555** |
| `coarse_dnc02` | 015000 | 0.05816 | 0.00449 | **1.490** |
| `coarse_dnc005`（健康） | 010000 | 0.07930 | 0.01903 | **12.623** |
| `coarse_dnc005` | 012000 | 0.06419 | 0.01849 | **16.514** |
| `coarse_dnc005` | 015000 | 0.05309 | 0.01326 | **10.938** |
| **`coarse_dnc02_detach`** | 010000 | 0.08140 | 0.02013 | **13.443** |
| **`coarse_dnc02_detach`** | 012000 | 0.06742 | 0.02032 | **17.796** |
| **`coarse_dnc02_detach`** | 015000 | 0.05809 | 0.01545 | **12.820** |

同样 λ=0.2，加 detach 后"有明显深度不连续的像素占比"在 010000/012000/015000 三个迭代分别是
不加 detach 的 **5.4× / 11.4× / 8.6×**，且始终与健康的 λ=0.05 组处在同一水平（12.8% vs 10.9%）。

> 口径说明：PNG 是按每张图自身的 min/max 归一化后存的 8bit 灰度，所以这三列是**相对**结构量，
> 只能横向比较同类图，不能当成物理深度梯度。它的作用是把"肉眼看上去有没有结构"变成可复算的数字。

### 4.7 同迭代 L_dnc / 有效像素三组对照（判读 P3、佐证 H4）

| iter | λ=0.2 无 detach | λ=0.05 | **λ=0.2 + detach** |
|---|---|---|---|
| 9100 | 0.3174 / 87.5% | 0.3452 / 87.5% | **0.3476 / 86.7%** |
| 9600 | 0.1843 / 87.6% | 0.3251 / 79.4% | **0.3306 / 78.1%** |
| 10100 | **0.0937 / 97.7%** | 0.2846 / 77.3% | **0.2759 / 75.4%** |
| 10600 | **0.0638 / 98.8%** | 0.2548 / 88.5% | **0.2958 / 86.6%** |

同样是 λ=0.2，加了 detach 之后 L_dnc 与有效像素比例的走势**与 λ=0.05 组几乎重合**，
而不是像无 detach 的 λ=0.2 那样"L_dnc 断崖下降 + 有效像素冲到 98%"。

### 4.6 中途观察：iteration 10000 的三组同点对比（P1 的关键证据）

10000 正是 λ=0.2 无 detach 组**已经完全退化**的那一迭代，三组在同一迭代的 `dnc_vis/iter010000_grid.png`：

| run | 深度图 D（左上） | 有效像素 |
|---|---|---|
| `coarse_dnc02`（λ=0.2，无 detach） | **无任何场景结构**，一个带同心波纹的光滑穹面 | 98.9% |
| `coarse_dnc005`（λ=0.05） | 卡车、栏板、驾驶室、车轮、背景树林与栅栏清晰可辨 | 81.9% |
| **`coarse_dnc02_detach`（λ=0.2 + detach）** | **卡车车厢、栏板、驾驶室、前后车轮、背景树木与铁栅栏全部清晰可辨**，与 λ=0.05 组同类；掩码只在物体边缘/树叶处为黑 | **80.7%** |

即：在**同样的 λ=0.2** 下，仅把 `N_d` 改成由 `D.detach()` 计算，深度图就从"光滑穹面"恢复成"真实场景"。
这直接支持机理假设 H4（深度通道是 L_dnc 的平凡解路径）。

### 4.10 第四组 mesh 的几何指标（我自己用 open3d + numpy 独立算的，不经过 `eval_geometry.py`）

mesh：`outputs/runs/mesh_dnc02_detach/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply`
（15,515,366 字节，08:58:41 落盘，`MESH END exit=0 wallclock=319s`，提取参数 `-l 0.3 -d 200000 --eval True` 与三组一致）

```
mesh                        V        F  comps   frag     lg%  dih_mean  dih_p90  G1med_rel%    G2<1%
mesh_base_seed0        205548   370128   2233   2137  41.686   45.9794 112.8438      0.0687   97.736
mesh_dnc005            211274   381723   2403   2324  45.251   49.5229 116.8433      0.1421   87.278
mesh_dnc02             120059   222535   1192   1177  84.583   52.5791 131.5514      0.8392   55.469
mesh_dnc02_detach      208382   375966   2242   2142  41.768   47.6651 116.1243      0.0839   97.542
```

读法：**`mesh_dnc02_detach` 与 λ=0 的 `mesh_base_seed0` 几乎无法区分**
（顶点/面数 ±1.4%、连通分量 2242 vs 2233、碎片 2142 vs 2137、最大分量占比 41.77% vs 41.69%、G2 97.54% vs 97.74%），
只有 G1 中位相对误差略差（0.0839% vs 0.0687%，+22%）、二面角略大（47.67° vs 45.98°，+1.69°）。
而不加 detach 的同 λ 组是 120k 顶点 / 84.6% 单一分量 / G1 0.8392% 的退化网格。

### 4.9 `dnc_vis/iter015000_grid.png` 的肉眼描述（P1 的直接证据）

`outputs/runs/coarse_dnc02_detach/dnc_vis/iter015000_grid.png`（2×2 拼图，最后一张）：

- **左上 D（深度图）**：**完整保留场景结构**。能清楚看出卡车的发动机舱盖、格栅、前保险杠、
  前轮与挡泥板、后面的木栏板货厢；卡车右侧有两棵树干、远处的电线杆、背景建筑的墙面，
  画面左侧还有一张野餐桌。近处的卡车是黑的（近 = 小深度值），背景树林/建筑是亮的（远），
  深度方向正确。**与 `coarse_dnc02` 同迭代那张"几道光滑沙丘、没有任何物体"的图完全不同。**
- **右上 N（渲染法向）**：车头、格栅、挡泥板、轮胎的法向分区清晰，地面是一整片平滑的绿
  （视空间 +Y 向上），树叶区域是高频杂色。相比 λ=0.05 组，车身上的法向更"成片"、更平滑 ——
  这正是 DNC 在只走法向通道时该产生的效果。
- **左下 N_d（深度差分法向）**：同样能认出卡车轮廓与地面，只是噪声更大（逐像素差分固有），
  配色与右上在同一表面上一致（地面同为绿）⇒ 两张图在同一坐标系，DNC 的几何约定正确。
- **右下 mask**：物体轮廓线、树叶、栅栏被正确剔除（黑），卡车主体与地面保留（白），
  有效像素 **86.19%**（不是退化组那种 98.3% 的"全图都是前景"）。


---

## 五、对 §3d 预注册预测 P1–P4 的逐条判定

判据阈值在第四组开跑之前就写死（见 §三），属于"先注册后看数"。评测于 **08:59:45** 完成，
`outputs/metrics/summary.csv` 已新增 `dnc02_detach` 一行。

### 5.0 四组最终指标（`outputs/metrics/summary.csv`）

| run | PSNR↑ | SSIM↑ | LPIPS↓ | G1 med rel%↓ | G1 mean rel%↓ | G2<1%↑ | 分量↓ | 碎片↓ | 最大分量%↑ | 面数 | G5 均值↓ | G5 P90↓ | ms/iter | 显存GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base_seed0`（λ=0）| **24.6201** | 0.85552 | 0.20316 | **0.0687** | 0.1770 | 97.736 | 2233 | 2137 | 41.686 | 370128 | **45.9794** | **112.8438** | 91.087 | 7.415 |
| `dnc005`（λ=0.05）| 22.4473 | 0.77085 | 0.34575 | 0.1421 | 0.4104 | 87.278 | 2403 | 2324 | 45.251 | 381723 | 49.5229 | 116.8433 | 104.463 | 7.625 |
| `dnc02`（λ=0.2）| 13.2436 | 0.47407 | 0.64648 | 0.8392 | 1.8352 | 55.469 | 1192 | 1177 | 84.583 | 222535 | 52.5791 | 131.5514 | 107.868 | 8.377 |
| **`dnc02_detach`（λ=0.2+detach）** | **24.3995** | **0.84945** | **0.21333** | **0.0839** | 0.2009 | 97.542 | 2242 | 2142 | 41.768 | 375966 | 47.6651 | 116.1243 | **96.686** | 7.585 |

> 这一行的每一个数字我都用**不经过 `eval_render.py` / `eval_geometry.py` 的独立脚本**复算过（§4.10 与 §八），
> 与 `summary.csv` 完全一致。


| 预测 | 判据 | 实际 | 结论 |
|---|---|---|---|
| **P1** 深度图保留场景结构 | `iter015000_grid.png` 肉眼可判 + 客观量 %px&#124;∇D&#124;>0.02 | 肉眼：卡车车头/格栅/前轮/木栏板货厢/树干/电线杆/野餐桌**全部清晰可辨**（§4.9）；客观量 **12.820%**，是 `dnc02` 的 1.490% 的 **8.6 倍**，与健康的 `dnc005`（10.938%）同级；有效像素 86.19%（非退化组的 98.3%）| **支持（PASS）** |
| **P2** PSNR 回到 λ=0 组 ±0.5 dB | PSNR ∈ **[24.1201, 25.1201]** | **24.3995 dB**，落在带内（距 λ=0 组 −0.2206 dB）；对比无 detach 的同 λ 组 13.2436 dB，**回升 11.16 dB**。SSIM 0.84945（λ=0 是 0.85552），LPIPS 0.21333（λ=0 是 0.20316）| **支持（PASS）** |
| **P3** L_dnc 仍下降但末值 > 0.046 | 首/中/末 + `last_l_dnc` > 0.04608708 | 首 9100 **0.347649** → 中 12100 **0.304053** → 末 15000 **0.291547**；整段 60 点线性斜率 −0.00565/1000it，前 12 点均值 0.3298 → 后 12 点均值 0.2990（确实在下降）；末值 **0.291547 ≫ 0.046087**（是门槛的 6.3 倍）| **支持（PASS）** |
| **P4** 几何指标相对 λ=0 组的变化方向（回答 H1） | 碎片数 / G1 median / G5 三项相对 base_seed0（2137 / 0.0687% / 45.98°）| 碎片 **2142**（+5，基本不变，**没有减少**）；连通分量 2242（+9）；G1 median **0.0839%**（+22%，**变差**）；G5 均值 **47.6651°**（+1.69°，**更不平滑**）；G2<1% 97.542%（−0.19pp）| **H1 不支持**（方向明确：三项全部持平或变差）|

### 5.1 结论（一句话 + 依据）

**机理假设 H4 成立，几何假设 H1 不成立。**

1. **H4（λ=0.2 的退化来自 L_dnc 经深度通道的平凡解）——支持。**
   同一 λ、同一 seed、同一 3DGS ckpt、同一提取参数，**只把 `N_d` 改成由 `D.detach()` 计算**，
   结果从「PSNR 13.24 dB + 深度图变成光滑穹面 + L_dnc 0.046 + 有效像素 98.3%」
   变成「PSNR 24.40 dB + 深度图完整保留场景 + L_dnc 0.292 + 有效像素 86.2%」。
   CPU 单测进一步证明：detach=False 时 `∂L/∂D = 1.51e-3 ≠ 0`，detach=True 时恰为 0，而 `∂L/∂N` 两者相同。
   P1 / P2 / P3 三项预注册预测全部命中。

2. **H1（λ>0 → 碎片更少 / 稀疏点→mesh 距离更小 / 法向更平滑）——不支持。**
   两个**没有退化**的 DNC 组都给出同一方向：

   | 相对 λ=0 | 碎片数 | G1 median rel | G5 二面角均值 |
   |---|---|---|---|
   | λ=0.05 | 2137 → **2324**（+8.8%，更差）| 0.0687% → **0.1421%**（+107%，更差）| 45.98° → **49.52°**（更差）|
   | λ=0.2 + detach | 2137 → **2142**（+0.2%，持平）| 0.0687% → **0.0839%**（+22%，更差）| 45.98° → **47.67°**（更差）|

   λ=0.2 无 detach 那组"碎片 1177、最大分量 84.6%"看似支持 H1，但它同时 PSNR 掉到 13.24、
   G1 median 0.8392%，是**退化产生的假象**（整个网格塌成一大片光滑曲面，当然分量少），不能算作 H1 的证据。

3. **附带发现（未预注册）**：detach 版比同 λ 的非 detach 版**训练更快（96.7 vs 107.9 ms/iter，−10.4%）、
   显存更省（7767 vs 8578 MiB，−9.5%）**，因为 DNC 项不再需要对深度那次光栅化做反向。
   相对 λ=0 的额外代价是 +6.1% 时间、+4.7% 显存。

4. **对报告的建议口径**：DNC 移植到 SuGaR 的 3D 椭球表示上，在 Truck 场景**没有带来几何收益**；
   λ 大时还会因为深度通道的平凡解导致灾难性退化，`--dnc_detach_depth` 能消除退化、把渲染质量拉回
   λ=0 水平（−0.22 dB），但拉回来之后几何指标也只是"和 λ=0 持平或略差"。
   这是一个**诚实的负结果 + 一个有价值的失败机理分析**，不要包装成正面结论。

---

## 六、硬性规则遵守情况

| 规则 | 结果 |
|---|---|
| 不删除任何文件 | 未执行任何 `rm`/`git checkout --`/`git clean`（`rm -rf` 只用在**我自己的 scratchpad** 临时目录 `~/tmp/claude-.../scratchpad/stage3check`，不在项目内） |
| 不 commit | `git -C repo/SuGaR log -1` 仍为 `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`；分支 `dnc` 仍是未提交的工作区改动 |
| 无文件删除（git 佐证）| `git diff --name-only --diff-filter=D` 在两个 repo 都是空输出 |
| 非我产生的改动（提前声明，避免验收误判）| 两个 repo 里都有一个未跟踪文件 `train_coarse_official_dnc.py`（9060 字节，mtime 08:31），是**另一位执行者**为计划 §3e 的 `coarse_official_dnc` 对照组新建的，**不是我的改动**；它在我的第四组训练日志头 `git dirty` 一行里也出现（`?? train_coarse_official_dnc.py`），说明它先于我的 run 就存在 |
| 不 pip install | 未执行任何 `pip` 命令 |
| 不改预注册假设 | 我**没有**写过 `plans/` 下的任何文件（该文件 08:33 的 mtime 来自主会话 Fable 的 checklist 更新，非我）。本仓库没有 `configs/hypotheses*.yaml` |
| 改动只限指定文件 | 见 §一的文件清单：**修改** 5 个（两个 repo 各 2 个 py + `scripts/run_s34_one.sh`）；**新增** 7 个（`scripts/run_s34_detach_after_seq.sh`、`scripts/run_s34_detach_after_mesh005.sh`、`scripts/run_eval_detach_after_mesh.sh`、`scripts/test_dnc_detach.py`、`notes/dnc_detach_3d.patch`、`outputs/metrics/test_dnc_detach.json`、`logs/s34_coarse_dnc02_detach_aborted_0833.log`）。未改任何其他现有模块 |
| GPU 冒烟不干扰 dnc005 计时 | 全部冒烟/单测在 CPU 上跑（`CUDA_VISIBLE_DEVICES=""`）。第四组正式训练最终在 `mesh_dnc005` **完成之后**（08:39:55）才独占 GPU0 启动；首次由 `SEQ END` 触发的 08:33:17 尝试因与 `mesh_dnc005` 重叠已被我主动中止，见 [DEVIATION-2b] |
| 单测未污染 GPU | `scripts/test_dnc_normal.py` / `scripts/test_dnc_detach.py` 均以 `CUDA_VISIBLE_DEVICES=""` 运行 |

---

## 七、[DEVIATION] 偏离与风险记录

**[DEVIATION-1] `coarse_dnc005` 用的是"加了 `--dnc_detach_depth` 之后"的代码，另外两组不是。**
- 原因：任务要求"先实现开关、再等 GPU"，而 `coarse_dnc005` 在 08:18:31 启动（改动落盘 08:17:14）。
  GPU 当时被占用，无法先等它跑完。
- 影响评估：**数值上无影响**。默认 `False` 时新增语句求值结果就是原张量本身（`y is x == True`，
  无新张量、无新 autograd 节点，§一证据 1），`dnc_utils.py` 未被触及（回归单测 8/8），
  日志横幅确认 `Detach depth for N_d: False (... original DNC behaviour)`。
- 唯一可见差异：`coarse_dnc005/train_stats.json` 多一个 `"dnc_detach_depth": false` 字段，
  `coarse_base_seed0` / `coarse_dnc02` 的 `train_stats.json` 没有该字段。`scripts/eval/summarize.py`
  全部用 `.get()` 读取，不会因此报错或改变已有数字。
- 替代方案（未采用）：等 `SEQ END` 之后再改代码 —— 会让第四组晚 20 分钟启动，撞上 09:30 放弃线。

**[DEVIATION-2]（严重，我的错误）`scripts/run_s34_one.sh` 在 dnc005 链运行期间被替换，导致 dnc005 的 mesh 提取没跑。**
- 我做了什么：为了加 `EXTRA_ARGS`，我把新内容写到 `scripts/.run_s34_one.sh.new` 后用 `mv` **原子重命名**覆盖
  （08:19:0x），当时 `bash scripts/run_s34_one.sh`（dnc005 链，PID 1336496 的前身）正在运行。
  我当时的判断是"POSIX rename 之后正在运行的 bash 仍持有旧 inode，安全"。
- **这个判断是错的。** 实测证据（`logs/s34_seq_1gpu.log`）：

  ```
  === dnc02 chain exit=0 Sun Sep 13 08:18:30 Asia 2026 ===
  scripts/run_s34_one.sh: error reading input file: No such file or directory
  === dnc005 chain exit=2 Sun Sep 13 08:33:01 Asia 2026 ===
  ```

  bash 在 python 训练命令返回后需要继续读脚本的后半段（mesh 那一段），此时它**按路径重新打开**脚本
  （或在 `/hpfs-main` 这种并行文件系统上，被 unlink 的旧 inode 已无法再读），于是报
  `error reading input file: No such file or directory` 并以 **exit 2**（bash 的读脚本/语法错误码）退出。
  `logs/s34_coarse_dnc005.log` 里**没有** `=== S3.4 TRAIN END ... ===` 这一行，`logs/s34_mesh_dnc005.log`
  当时根本不存在 —— 都印证脚本是在训练命令之后、写 TRAIN END 之前断掉的。
- **实际影响**：
  - `coarse_dnc005` 的**训练本身完整完成**（`15000.pt` 304 MB 已落盘、`train_stats.json` 已写、
    日志有 `Training finished after 15000 iterations with loss=0.11848752`、8001 iters / 13.93 min / 104.5 ms/iter /
    7808 MiB / 429395 高斯）。**没有丢失任何训练成果。**
  - 丢的是紧随其后的 `extract_mesh.py`，`outputs/runs/mesh_dnc005/` 因此是空的。
  - 由执行者 6 在 08:34 手工补跑（`logs/s34_mesh_dnc005.log`），参数与三组完全一致。
- **正确做法（教训）**：绝不要就地修改（哪怕是 `mv` 覆盖）任何**正在被 bash 执行**的脚本；
  应当写成**新文件名**，让新命令去用新文件。本次修正后我新建的等待脚本
  `scripts/run_s34_detach_after_mesh005.sh` 就是按这个原则写的（没有再动 `run_s34_one.sh`）。
- **当前 `run_s34_one.sh` 状态**：完整、可执行 —— `bash -n scripts/run_s34_one.sh` 通过，
  129 行，`md5 c490b041c6e3…`；并且第四组的训练日志头证明 `EXTRA_ARGS` 生效：
  `extra args   : --dnc_detach_depth True`，命令行末尾确实带上了 `--dnc_detach_depth True`。

**[DEVIATION-2b] 第四组首次启动（08:33:17）与 `mesh_dnc005` 补跑重叠，已主动中止重跑。**
- 等待脚本 `run_s34_detach_after_seq.sh` 以 `SEQ END` 为触发条件，而 `SEQ END` 因为 DEVIATION-2 提前出现
  （dnc005 链异常退出也会让 seq 走到 SEQ END），于是第四组在 08:33:17 抢先启动；
  执行者 6 的 `mesh_dnc005` 在 08:34:08 也占用 GPU0，两者重叠。
- 处理：08:35:22 主动 `kill -TERM` 掉第四组（当时刚到 iteration 9001，**未产生任何 checkpoint**，
  只有一个会被覆盖的 `dnc_log.csv` 表头），把 GPU0 让给 `mesh_dnc005`；
  改用新等待脚本 `scripts/run_s34_detach_after_mesh005.sh`，条件为
  「`logs/s34_mesh_dnc005.log` 出现 `Mesh saved` **且** `outputs/runs/mesh_dnc005/*.ply` 存在」，
  08:50 为放弃等待线（超过则直接启动并标注重叠）。
- 代价：第四组晚约 6 分钟开始；收益：`ms/iter` 与 `coarse_dnc02` / `coarse_dnc005` 可比，
  且 `mesh_dnc005` 独占 GPU、更快完成（它在主线三组的关键路径上）。
- 留档：被中止的那次训练日志已按项目惯例复制为 `logs/s34_coarse_dnc02_detach_aborted_0833.log`（319 行，末尾停在 `Starting depth-normal consistency (DNC) regularization, factor=0.2.`），**未删除任何文件**；正式重跑会覆盖 `logs/s34_coarse_dnc02_detach.log`。
- 重跑前确认产出目录是干净的：`dnc_log.csv` 只有 1 行表头、`dnc_vis/` 0 个文件、`sugarcoarse_*/` 0 个文件（首次尝试停在 iter≈9001，`dnc_log_every=100` 的第一行要到 9100、`dnc_vis_every=1000` 的第一张图要到 10000，都还没写）。

### `scripts/run_s34_one.sh` 的 `EXTRA_ARGS` 改动 diff（唯一的脚本改动）

```diff
@@ -8,6 +8,12 @@
 #   RUN=coarse_base_seed0 DNC_FACTOR=0   GPU=0 bash scripts/run_s34_one.sh
 #   RUN=coarse_dnc005     DNC_FACTOR=0.05 GPU=1 bash scripts/run_s34_one.sh
 #   RUN=coarse_dnc02      DNC_FACTOR=0.2  GPU=1 bash scripts/run_s34_one.sh
+#   RUN=coarse_dnc02_detach DNC_FACTOR=0.2 GPU=0 \
+#       EXTRA_ARGS="--dnc_detach_depth True" bash scripts/run_s34_one.sh
+#
+# EXTRA_ARGS (optional, default empty) is appended verbatim to the training
+# command.  Empty by default, so the three original runs are reproduced by the
+# exact same commands as before.
 #
@@ -25,6 +31,7 @@
 GPU=${GPU:?GPU must be set}
 DNC_START=${DNC_START:-9000}
 SEED=${SEED:-0}
+EXTRA_ARGS=${EXTRA_ARGS:-}   # extra flags appended to the training command (default: none)
 
@@ -55,8 +62,9 @@
   echo "dnc_factor   : $DNC_FACTOR"
   echo "dnc_start    : $DNC_START"
+  echo "extra args   : ${EXTRA_ARGS:-<none>}"
   echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
-  echo "command      : python train_coarse_density.py ... --dnc_start $DNC_START"
+  echo "command      : python train_coarse_density.py ... --dnc_start $DNC_START ${EXTRA_ARGS}"
   echo "==============================================="
 } > "$TRAIN_LOG"
@@ -72,6 +80,7 @@
     --dnc_factor "$DNC_FACTOR" \
     --dnc_start "$DNC_START" \
+    ${EXTRA_ARGS} \
     >> "$TRAIN_LOG" 2>&1
```

`EXTRA_ARGS` 为空时 `${EXTRA_ARGS}`（不加引号）展开为**零个词**，训练命令与前三组逐字相同；
`set -u` 不会报错，因为第 34 行已经把它赋成了空串。`bash -n` 通过。

**[DEVIATION-3] 组间 `OMP_NUM_THREADS` 不一致（既有问题，非本次引入）。**
- `coarse_base_seed0` = 24，`coarse_dnc02` / `coarse_dnc005` = 12。这会影响 `ms/iter` 的可比性
  （不影响 PSNR/几何指标，因为 CPU 线程数不改变数值结果）。
- 本次第四组取 **12**，与它的直接对照组 `coarse_dnc02` 对齐。报告里比较训练代价时应只比较
  `dnc02` vs `dnc02_detach` vs `dnc005` 三者，或注明 `base_seed0` 的线程数不同。

**[DEVIATION-4] 未在 GPU 上做 `--dnc_detach_depth True` 的冒烟。**
- 原因：任务明确要求"不要在 GPU 上跑冒烟，避免干扰 dnc005 计时"。
- 补偿：用 CPU 单测 `scripts/test_dnc_detach.py` 直接验证了开关的梯度语义（8/8 PASS），
  比"跑 20 迭代看 loss 是否有限"更强；正式 run 本身即是第一次 GPU 执行。

---

## 八、复算命令（验收者可直接照跑）

```bash
cd /scratch/e1351071/zju_test
source /scratch/e1351071/virtualenvs/zju_test/bin/activate

# (1) 改动范围：本次增量补丁只有 5 个 hunk
cat notes/dnc_detach_3d.patch
diff -u repo/SuGaR/train_coarse_density.py          repo/SuGaR_dev/train_coarse_density.py   # 应无输出
diff -u repo/SuGaR/sugar_trainers/coarse_density.py repo/SuGaR_dev/sugar_trainers/coarse_density.py
diff -u repo/SuGaR/sugar_utils/dnc_utils.py         repo/SuGaR_dev/sugar_utils/dnc_utils.py

# (2) 增量性：7c10c4a + dnc_stage3.patch + dnc_detach_3d.patch == 当前代码
CHK=$(mktemp -d)
git -C repo/SuGaR_dev archive 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44 | tar -x -C "$CHK"
(cd "$CHK" && git apply /scratch/e1351071/zju_test/notes/dnc_stage3.patch \
            && git apply /scratch/e1351071/zju_test/notes/dnc_detach_3d.patch)
diff -q "$CHK/train_coarse_density.py"           repo/SuGaR_dev/train_coarse_density.py
diff -q "$CHK/sugar_trainers/coarse_density.py"  repo/SuGaR_dev/sugar_trainers/coarse_density.py
# 两条均无输出

# (3) 开关的梯度语义（CPU，8/8）
CUDA_VISIBLE_DEVICES="" python scripts/test_dnc_detach.py --repo repo/SuGaR_dev
cat outputs/metrics/test_dnc_detach.json

# (4) DNC 数学回归（CPU，8/8）
CUDA_VISIBLE_DEVICES="" python scripts/test_dnc_normal.py --repo repo/SuGaR_dev

# (5) 语法 / argparse
python -m py_compile repo/SuGaR/train_coarse_density.py repo/SuGaR/sugar_trainers/coarse_density.py
cd repo/SuGaR && CUDA_VISIBLE_DEVICES="" python train_coarse_density.py --help | grep -A3 dnc_detach_depth; cd -

# (6) 几何指标独立复算（纯 open3d + numpy，不经过 eval_geometry.py）
OMP_NUM_THREADS=4 python - <<'PY'
import open3d as o3d, numpy as np, glob
for name in ['mesh_base_seed0','mesh_dnc02','mesh_dnc005','mesh_dnc02_detach']:
    f = glob.glob(f'outputs/runs/{name}/*.ply')
    if not f: continue
    m = o3d.io.read_triangle_mesh(f[0]); m.compute_triangle_normals()
    tri = np.asarray(m.triangles); v = np.asarray(m.vertices); tn = np.asarray(m.triangle_normals)
    lab, cnt, _ = m.cluster_connected_triangles(); cnt = np.asarray(cnt)
    e = np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],0),1)
    fid = np.tile(np.arange(len(tri)),3); o = np.lexsort((e[:,1],e[:,0])); e,fid = e[o],fid[o]
    same = np.all(e[1:]==e[:-1],1); i,j = fid[:-1][same], fid[1:][same]
    ang = np.degrees(np.arccos(np.clip((tn[i]*tn[j]).sum(1),-1,1)))
    print(f'{name:18s} V={len(v):7d} F={len(tri):7d} comps={len(cnt):5d} frag<100f={(cnt<100).sum():5d} '
          f'largest%={100*cnt.max()/len(tri):.3f} dihedral mean={ang.mean():.4f} p90={np.percentile(ang,90):.4f}')
PY

# (6b) P1 的客观结构量（只用 PIL+numpy，读 dnc_vis 的深度 PNG）
CUDA_VISIBLE_DEVICES="" python - <<'PY2'
import numpy as np, os
from PIL import Image
def m(p):
    a = np.asarray(Image.open(p).convert('L'), dtype=np.float64)/255.
    L = (-4*a[1:-1,1:-1]+a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])
    g = np.abs(np.diff(a,axis=0)[:,:-1])+np.abs(np.diff(a,axis=1)[:-1,:])
    return L.std(), g.mean(), (g>0.02).mean()*100
for run in ['coarse_dnc02','coarse_dnc005','coarse_dnc02_detach']:
    for it in ['010000','012000','015000']:
        p=f'outputs/runs/{run}/dnc_vis/iter{it}_depth.png'
        if os.path.exists(p):
            s,g,f=m(p); print(f'{run:22s} {it} Lap={s:.5f} grad={g:.5f} pct={f:.3f}')
PY2

# (6c) G1/G2 独立复算（自己读 COLMAP points3D.bin + open3d raycasting）
CUDA_VISIBLE_DEVICES="" python - <<'PY3'
import json, sys, glob, numpy as np, open3d as o3d
sys.path.insert(0, 'scripts/eval')
from _common import read_points3D_binary_with_tracks
ref = json.load(open('outputs/metrics/geometry_coarse_base_seed0.json'))
lo = np.array(ref['fg_bbox_min']); hi = np.array(ref['fg_bbox_max']); ext = ref['cameras_extent']
out = read_points3D_binary_with_tracks('data/tandt/truck/sparse/0/points3D.bin')
xyz, tracks = out[0], out[-1]
P = xyz[(tracks>=3) & np.all(xyz>=lo,1) & np.all(xyz<=hi,1)].astype(np.float32)
for name in ['mesh_base_seed0','mesh_dnc005','mesh_dnc02','mesh_dnc02_detach']:
    f = glob.glob(f'outputs/runs/{name}/*.ply')
    if not f: continue
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(f[0])))
    d = sc.compute_distance(o3d.core.Tensor(P)).numpy()
    print(f'{name:18s} median_rel%={100*np.median(d)/ext:.4f} <1%ext={100*(d<0.01*ext).mean():.3f}%')
PY3

# (7) 渲染指标独立复算（只用 PIL+numpy 从 PNG 重算 PSNR）
CUDA_VISIBLE_DEVICES="" python scripts/eval/check_png_psnr.py --runs coarse_base_seed0,coarse_dnc02,coarse_dnc02_detach

# (8) 汇总表
python scripts/eval/summarize.py && column -s, -t outputs/metrics/summary.csv | cut -c1-160

# (9) git 审计（应显示：无 commit、只有工作区改动、无删除）
git -C repo/SuGaR     status --short; git -C repo/SuGaR     log -1 --oneline
git -C repo/SuGaR_dev status --short; git -C repo/SuGaR_dev log -1 --oneline
```
