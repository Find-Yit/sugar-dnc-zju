# SELF_CHECK —— 阶段 3 / S3.4 三组对照（C3、C4）

执行者 6，2026-09-13 08:45。计划：`plans/SuGaR复现与改动_6小时考核.txt` 阶段 3 的 C3 / C4。
本次**未修改任何代码、未删除任何文件、未 commit、未 pip install、未启动额外训练**；GPU 只用于
`scripts/eval/` 的评测脚本与 `coarse_dnc005` 的 mesh 补跑（补跑原因见 `notes/RUNLOG_s34.md` §4.4）。

三组正式对照 run：`coarse_base_seed0`(λ=0) / `coarse_dnc005`(λ=0.05) / `coarse_dnc02`(λ=0.2)。
另有阶段 2 未打补丁的 `coarse_baseline` 作留档行。

---

## C3 三组 coarse `.pt` 与 mesh `.ply` 齐全、墙钟齐全、λ 组有 L_dnc 数值

### C3.1 产物齐全

| run | `15000.pt` | 字节 | mesh `.ply` | 字节 | 结果 |
|---|---|---|---|---|---|
| `coarse_base_seed0` | `outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt` | 304,039,190 | `outputs/runs/mesh_base_seed0/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` | 15,294,938 | PASS |
| `coarse_dnc005` | `outputs/runs/coarse_dnc005/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt` | 304,019,606 | `outputs/runs/mesh_dnc005/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` | 15,737,699 | PASS |
| `coarse_dnc02` | `outputs/runs/coarse_dnc02/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt` | 304,034,390 | `outputs/runs/mesh_dnc02/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` | 9,016,290 | PASS |

**每个 `.ply` 用 open3d 独立读出（不依赖评测脚本，脚本见本文件末"复算命令"）**：

| run | V（顶点） | F（面） | NaN 顶点 | Inf 顶点 | 越界面索引 | 顶点色/法向 | 结果 |
|---|---|---|---|---|---|---|---|
| `coarse_base_seed0` | 205,548 | 370,128 | 0 | 0 | 0 | 有 / 有 | PASS |
| `coarse_dnc005` | 211,274 | 381,723 | 0 | 0 | 0 | 有 / 有 | PASS |
| `coarse_dnc02` | 120,059 | 222,535 | 0 | 0 | 0 | 有 / 有 | PASS |
| （留档）`coarse_baseline` | 205,647 | 370,438 | 0 | 0 | 0 | 有 / 有 | PASS |

包围盒：base_seed0 `[-23.398,-15.636,-23.336]~[23.566,6.633,23.707]`、
dnc005 `[-23.486,-15.651,-23.336]~[23.592,6.182,23.789]`、
dnc02 `[-9.267,-3.546,-9.130]~[9.718,7.101,11.961]`。
**注意 dnc02 的包围盒明显收缩、面数只有另两组的 60%**，与其渲染崩塌是同一现象（见 §C3.3）。

### C3.2 训练代价关键字段（全部来自各 run 的 `train_stats.json`，未手抄）

| 字段 | `coarse_base_seed0` (λ=0) | `coarse_dnc005` (λ=0.05) | `coarse_dnc02` (λ=0.2) |
|---|---|---|---|
| `train_wallclock_min`（训练循环净时间） | **12.146** | **13.930** | **14.384** |
| 脚本 date 打点墙钟（含加载/存盘） | 779 s = 12 m 59 s | — (chain exit=2 未打点，train_stats 记 862.8 s 总时) | 911 s = 15 m 11 s |
| `mean_time_per_iteration_ms` | **91.087** | **104.463** | **107.868** |
| `max_memory_allocated_MiB` | **7,593.42** | **7,807.54** | **8,578.43** |
| 同上换算 GiB | 7.415 | 7.625 | 8.377 |
| `max_memory_reserved_MiB` | 9,110.0 | 9,110.0 | 16,648.0 |
| `n_gaussians_final` | **429,423** | **429,395** | **429,416** |
| `n_iterations_done` / `last_iteration` | 8001 / 15000 | 8001 / 15000 | 8001 / 15000 |
| `final_loss` | 0.083459 | 0.118488 | 0.261579 |
| mesh 提取墙钟 | 416 s | 346 s（手动补跑） | 334 s |

> 相对 λ=0：λ=0.05 的每迭代耗时 +14.7%、峰值显存 +2.8%；λ=0.2 的每迭代耗时 +18.4%、峰值显存 +13.0%。
> **这两个百分比不是严格受控**：`OMP_NUM_THREADS` 与运行节点在组间不同（RUNLOG §4.5 DEV-1/DEV-2/DEV-4），
> 报告里只能作量级参考。

### C3.3 λ 组 `dnc_log.csv` 的 L_dnc 与 `sdf_better_normal_loss` 走势（H3 证据）

两组 CSV 均为 **60 行数据（iter 9100 → 15000，每 100 iter 一行）+ 表头**，无缺行、无 NaN。

| 量 | run | 首值 (it 9100) | 中值 (it 12100) | 末值 (it 15000) | 首 5 点均值 | 末 5 点均值 | 线性斜率 /1000 iter | min / max |
|---|---|---|---|---|---|---|---|---|
| `l_dnc` | dnc005 | 0.34524450 | 0.25968462 | **0.23466052** | 0.33766375 | 0.24029247 | **−0.01264** | 0.19733 / 0.37294 |
| `l_dnc` | dnc02 | 0.31742537 | 0.05566579 | **0.04608708** | 0.24575392 | 0.05942887 | **−0.01893** | 0.02484 / 0.31743 |
| `sdf_better_normal_loss` | dnc005 | 0.09267963 | 0.03106488 | **0.02307408** | 0.07119072 | 0.02391350 | **−0.00658** | 0.02123 / 0.09268 |
| `sdf_better_normal_loss` | dnc02 | 0.09766620 | 0.02379805 | **0.02010683** | 0.07414518 | 0.01962919 | **−0.00729** | 0.01689 / 0.09767 |
| `sdf_estimation_loss` | dnc005 | 0.33315569 | 0.21995579 | 0.17765826 | 0.30689746 | 0.17550620 | −0.02059 | 0.16803 / 0.33316 |
| `sdf_estimation_loss` | dnc02 | 0.36093274 | 0.18717416 | 0.11605667 | 0.33349503 | 0.11621304 | −0.03511 | 0.10644 / 0.36093 |
| `valid_pixel_ratio` | dnc005 | 0.874919 | 0.848483 | 0.878298 | 0.824476 | 0.865892 | +0.00438 | 0.73646 / 0.93503 |
| `valid_pixel_ratio` | dnc02 | 0.875091 | 0.986542 | 0.983241 | 0.859373 | 0.985724 | +0.01191 | 0.79336 / 0.98862 |

**判定**：两组的 `l_dnc` 与 `sdf_better_normal_loss` 都在下降（斜率为负，末 5 点均值 < 首 5 点均值），
**没有出现"一个降另一个升"的冲突**。λ=0.05 的 L_dnc 只从 0.338 降到 0.240（−29%），
λ=0.2 从 0.246 降到 0.059（−76%）——权重越大压得越狠。
对照组 `coarse_base_seed0` 的 `train_stats.json` 里 `last_l_dnc = null`、`dnc_enabled = false`、
训练日志中 **0 行** `[DNC]`，证明 λ=0 完全不进入 DNC 分支（对应计划 S3.5）。

**C3 结论：PASS**（产物齐全、墙钟齐全、λ 组 L_dnc 有数值且下降）。

---

## C4 三组用同一 3DGS ckpt、同一 seed、同一 extract 参数

| 核对项 | `coarse_base_seed0` | `coarse_dnc005` | `coarse_dnc02` | 一致？ |
|---|---|---|---|---|
| 场景 `-s` | `data/tandt/truck` | 同 | 同 | ✅ |
| 3DGS ckpt `-c` | `outputs/baseline/gs_truck/` | 同 | 同 | ✅ |
| 3DGS ckpt 内容 md5（`point_cloud/iteration_7000/point_cloud.ply`） | `335593595301383a2780b0a790e9b4cf`（评测后复核，全程未变） | — | — | ✅ |
| 载入迭代 `-i` | 7000 | 7000 | 7000 | ✅ |
| `--eval` | True（llffhold=8 → 219 train / 32 test） | 同 | 同 | ✅ |
| `--seed` | 0 | 0 | 0 | ✅ |
| 日志里 `[REPRO] random / numpy / torch seeded with 0.` | 有（1 行） | 有（1 行） | 有（1 行） | ✅ |
| `--dnc_start` | 9000 | 9000 | 9000 | ✅ |
| `-e/--estimation_factor`（`sdf_estimation_factor`） | 0.2 | 0.2 | 0.2 | ✅ |
| `-n/--normal_factor`（`sdf_better_normal_factor`） | 0.2 | 0.2 | 0.2 | ✅ |
| 总迭代 `num_iterations` | 15000 | 15000 | 15000 | ✅ |
| mesh `-l`（surface_level） | 0.3 | 0.3 | 0.3 | ✅ |
| mesh `-d`（decimation target） | 200000 | 200000 | 200000 | ✅ |
| mesh `--eval` | True | True | True | ✅ |
| 唯一差异量 `--dnc_factor` | **0** | **0.05** | **0.2** | ✅（设计如此） |
| git base commit | `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44` | 同 | 同 | ✅ |
| 工作树 dirty 列表 | `M coarse_density.py / A dnc_utils.py / M train_coarse_density.py` | 同 | 同 | ✅（内容见下） |

证据：三条训练命令与三条 mesh 命令逐字见 `notes/RUNLOG_s34.md` §2 与 §4.2，
也可直接 `grep -h "^command      :" logs/s34_coarse_*.log logs/s34_mesh_*.log` 复核（本自检即如此取证）。

### C4 的一个**不合格项**（必须写进报告）

**DEV-3：三组并非在字节一致的工作树上运行。**
`repo/SuGaR_dev/train_coarse_density.py` 与 `sugar_trainers/coarse_density.py` 的 mtime = **08:17:14**，
落在 `coarse_dnc02` 启动（07:57:45）之后、`coarse_dnc005` 启动（08:18:31）之前；改动是另一位执行者
新增 `--dnc_detach_depth`（默认 False）。

- 直接证据：`logs/s34_coarse_dnc005.log:32` 有
  `[DNC] Detach depth for N_d: False (gradient through both D and N - original DNC behaviour)`，
  而 `logs/s34_coarse_dnc02.log` 与 `logs/s34_coarse_base_seed0.log` **没有这一行**；
  `coarse_dnc005/train_stats.json` 比另两组多一个 `"dnc_detach_depth": false` 字段。
- 影响评估：改动处是
  `dnc_depth_for_loss = dnc_depth.detach() if dnc_detach_depth else dnc_depth`（`coarse_density.py:865`），
  **`False` 分支与改动前逐字等价**，其余是一行 CONSOLE 打印和一个统计字段。
  旁证：被杀的旧 `coarse_dnc005`（06:16 启动，跑的是**改动前**的代码）在 it 9100 的
  `l_dnc = 0.34662309`，本次（改动后）同一迭代 `l_dnc = 0.34524450`，相对差 **0.40%**，
  落在 CUDA 光栅化非确定性的量级内。
- 结论：**功能上可视为同一实现，但"三组字节一致"这一条严格意义上 FAIL**，如实记录，不做粉饰。

**C4 结论：主要条目全部 PASS；DEV-3 为不合格项（影响评估为语义空操作，附旁证）。**

---

## 复算命令

```bash
source /scratch/e1351071/zju_test/env.sh
cd /scratch/e1351071/zju_test

# (1) 三组 .pt / .ply 是否存在与大小
ls -la outputs/runs/coarse_{base_seed0,dnc005,dnc02}/sugarcoarse_*/15000.pt \
       outputs/runs/mesh_{base_seed0,dnc005,dnc02}/*.ply

# (2) 用 open3d 独立读 mesh（V/F/NaN/越界面索引/包围盒）
python - <<'PY'
import glob, numpy as np, open3d as o3d
for p in sorted(glob.glob('outputs/runs/mesh_*/[!.]*.ply')):
    m = o3d.io.read_triangle_mesh(p); v = np.asarray(m.vertices); f = np.asarray(m.triangles)
    print(p, 'V=%d F=%d NaN=%d OOR=%d' % (len(v), len(f),
          int(np.isnan(v).any(1).sum()), int(((f<0)|(f>=len(v))).any(1).sum())))
PY

# (3) train_stats 关键字段
python - <<'PY'
import json
for r in ['coarse_base_seed0','coarse_dnc005','coarse_dnc02']:
    d = json.load(open(f'outputs/runs/{r}/train_stats.json'))
    print(r, {k: d[k] for k in ['train_wallclock_min','mean_time_per_iteration_ms',
              'max_memory_allocated_MiB','n_gaussians_final','last_iteration','dnc_factor','last_l_dnc']})
PY

# (4) dnc_log.csv 的首/中/末值与斜率
python - <<'PY'
import csv, numpy as np
for p in ['outputs/runs/coarse_dnc005/dnc_log.csv','outputs/runs/coarse_dnc02/dnc_log.csv']:
    rows = list(csv.DictReader(open(p))); it = np.array([int(r['iteration']) for r in rows])
    for k in ['l_dnc','sdf_better_normal_loss','sdf_estimation_loss','valid_pixel_ratio']:
        a = np.array([float(r[k]) for r in rows])
        print(p, k, 'first=%.8f mid=%.8f last=%.8f slope/1000it=%+.8f'
              % (a[0], a[len(a)//2], a[-1], np.polyfit(it, a, 1)[0]*1000))
PY

# (5) λ=0 不进 DNC 分支
grep -c "^\[DNC\]" logs/s34_coarse_base_seed0.log    # 期望 0

# (6) 参数一致性
grep -h "^command      :" logs/s34_coarse_*.log logs/s34_mesh_*.log | grep -v killed

# (7) DEV-3 的证据
ls -la --time-style=+%H:%M:%S repo/SuGaR_dev/train_coarse_density.py repo/SuGaR_dev/sugar_trainers/coarse_density.py
grep -n "Detach depth for N_d" logs/s34_coarse_dnc005.log logs/s34_coarse_dnc02.log
```
