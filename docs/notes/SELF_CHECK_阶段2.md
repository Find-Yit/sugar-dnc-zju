# SELF_CHECK — 阶段 2：基线流水线（3DGS 7k → coarse SuGaR 15k → Poisson mesh）

计划：`plans/SuGaR复现与改动_6小时考核.txt` §3「阶段 2」验收条目 B1–B5
执行：opus-executor　　时间：2026-09-13 05:48:58 → 06:11:41（**总墙钟 22m43s**，预算 50 min）
场景：Tanks&Temples / Truck，251 张 979×546，`--eval` + llffhold=8 → **train 219 / test 32**
SuGaR commit：`7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`（工作区 `git status --porcelain` 为空，**未改动任何仓库文件**）

| 条目 | 要求 | 实际值 | 判定 |
|---|---|---|---|
| B1 | `point_cloud/iteration_7000/point_cloud.ply` 存在；3DGS 日志末尾有 7000 iter 的 test PSNR | 文件存在，460,021,692 B，1,854,920 个高斯。日志：`[ITER 7000] Evaluating test: L1 0.03830549860140309 PSNR 23.908734679222107`。**独立复算**（`scripts/recompute_gs_psnr.py`，重新加载 ckpt 渲染全部 32 张 test）得 **PSNR 23.908734679222107 / L1 0.038305**，与日志**逐位一致** | **PASS** |
| B2 | coarse 输出 `.pt` 存在；日志显示到 15000 iter；出现 "Starting SDF regularization" | `.../coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`，303,793,046 B，**429,075 个高斯**（`_points (429075,3)`）。日志第 288 行 `Starting SDF regularization.`（另有 `Starting SDF estimation loss.`、`Starting SDF better normal loss.`）。末尾 `Training finished after 15000 iterations with loss=0.0918252244591713.` + `Final model saved.` | **PASS** |
| B3 | mesh `.ply` 存在；open3d 独立读取：顶点数 >0、三角形数 ≈ 200k（decimation 目标）、无 NaN | 文件 `sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply`，15,304,017 B。open3d 独立读取：**顶点 205,647 / 三角形 370,438**；`vertices_finite=True`、`any_nan_vertices=False`；带顶点色与法向。**三角形数与计划预期值 200k 不符，原因见下方 [DEVIATION-4]（是计划对上游参数语义的误解，数值本身正确）** | **PASS（附说明）** |
| B4 | `outputs/baseline/` 下留档三步产物索引（路径清单） | `outputs/baseline/INDEX.md`：三步全部产物绝对路径、字节数、高斯/顶点/面数、日志与脚本路径、关键指标 | **PASS** |
| B5 | 三步总墙钟记录在 `RUNLOG.md` | `notes/RUNLOG.md` §S2.4 表格：2m13s + 12m44s + 6m22s = **21m19s**（含失败重启间隙则 22m43s，05:48:58→06:11:41） | **PASS** |

**阶段 2 结论：B1–B5 全部 PASS（B3 附语义说明）。**

---

## 关键数字

### S2.1 3DGS 7000 iter（GPU0，2m13s，约 85 it/s）
| 指标 | 值 |
|---|---|
| SfM 初始点数 | 136,029 |
| 7000 iter 后高斯数 | 1,854,920 |
| **test PSNR（32 视角）** | **23.908735**（L1 0.038305）；单视角范围 20.9483 – 25.7173 |
| train PSNR（日志，5 个采样相机） | 24.764399 |
| train PSNR（复算，全部 219 视角） | 24.800641（L1 0.034118） |

> 日志里的 train PSNR 只在 5 个相机上算（`train.py:167` 取 `range(5,30,5)`），且训练时 `Scene(shuffle=True)` 打乱过顺序，
> 因此与"全部 219 视角"的复算值不可能逐位相同（25.5011 vs 24.7644 vs 24.8006 的差异来源即此）。
> **test 分支用的是全部 32 个 test 相机，故可逐位复现，已作为 B1 的独立证据。**

### S2.2 coarse SuGaR（GPU0，12m44s）
| 指标 | 值 |
|---|---|
| 训练区间 | 7000 → 15000 iter（entropy 正则 7000–9000；iter 9000 硬剪枝 opacity<0.5；SDF/density 正则 9000 起） |
| 剪枝后高斯数 | 429,075 |
| 最终 loss | 0.0918252244591713 |
| 末次 SDF 采样使用的高斯数 | 140,831 |
| 速度 | entropy 阶段 ~0.05 min/200 iter；SDF 阶段 ~0.37 min/200 iter |
| 峰值显存 / 利用率 | 约 10.1 GB / 96–100% |
| 正则超参 | `estimation_factor=0.2`、`normal_factor=0.2`（脚本默认，未改） |

### S2.3 mesh 提取（GPU0，6m22s）
| 指标 | 值 |
|---|---|
| surface level / decimation target | 0.3 / 200,000 |
| Poisson 原始前景网格 | 1,664,964 点 / 3,271,742 面 |
| Poisson 原始背景网格 | 2,289,399 点 / 4,500,788 面 |
| **最终（decimate + clean + project + merge）** | **205,647 顶点 / 370,438 面** |
| 连通分量数 | 2,237（最大分量 157,415 面，占 42.49%） |
| 面数 <100 的碎片分量 | **2,147 个** |
| 流形性 | edge-manifold True，vertex-manifold False |
| 包围盒 | [-23.487, -15.537, -23.317] ~ [23.602, 6.339, 23.712] |

> 碎片分量数 2,147 是阶段 3 预注册假设 **H1**（λ>0 时漂浮小连通分量数 ↓）的 baseline 参照值。

---

## 偏离与说明（[DEVIATION]）

**[DEVIATION-4] B3 的"三角形数 ≈ 200k"与上游语义不符 —— 正确预期应为 ≈2×200k**
- 计划 B3 写"三角形数 ≈ 200k（decimation 目标）"。实测 370,438 面。
- 根因（已核对上游源码，非实现问题）：
  1. `extract_mesh.py -d` 的 argparse help 写的是 "Target number of **vertices**"，但 `sugar_extractors/coarse_mesh.py:442/449` 把它直接传给
     `o3d.geometry.TriangleMesh.simplify_quadric_decimation(decimation_target)`，该函数的第一个位置参数是
     **`target_number_of_triangles`**（已用 `print(o3d...simplify_quadric_decimation.__doc__)` 确认）。即上游 help 文本本身就有误导。
  2. 更关键：decimation 对**前景网格和背景网格各做一次**，然后 `decimated_o3d_fg_mesh + decimated_o3d_bg_mesh` 合并
     （`coarse_mesh.py:437–470`）。所以 `-d 200000` 的合理预期是 **≈400k 面**，再被 clean/project 去掉一部分 → 370,438。
- 结论：**数值 370,438 面是正确且符合上游行为的**，与计划预期值的差异来自计划对参数语义的误解。
  顶点数 205,647 恰好 ≈200k 属巧合（顶点与面近似 1:2 关系下的自然结果），不应作为判据。
- 建议：阶段 3/4 三组对照仍统一用 `-l 0.3 -d 200000`，**对照组之间可比**即可；报告中按"面数 370k 量级"如实描述。

**[DEVIATION-5] S2.2 首次启动失败：`-c` 路径必须以 `/` 结尾（计划命令未带）**
- 报错：`FileNotFoundError: .../outputs/baseline/gs_truckcameras.json`，来自 `sugar_scene/cameras.py:34` 的
  `open(gs_output_path + 'cameras.json')`（字符串拼接，非 `os.path.join`）。
- 这是上游约定而非 bug：官方 `train_full_pipeline.py:130–131` 显式做
  `if gs_checkpoint_dir[-1] != os.path.sep: gs_checkpoint_dir += os.path.sep`。
- 处理：**只改我自己的运行脚本命令行**（`-c .../gs_truck/`），未改动仓库任何文件。失败日志留档
  `logs/s22_coarse_baseline_FAILED_nopathsep.log`（未删除）。损失墙钟约 1 分钟。
- 影响阶段 3：三组对照的 `-c` 均须带尾斜杠。

**[DEVIATION-6] 按主会话指示，S2.2 未传 `--dnc_factor`**
- 计划 S2.2 写了 `--dnc_factor 0`（并自注"阶段 3 加入前先不传"）。本次按主会话明确指示**完全不传该参数**，
  因此 baseline 走的是**未经任何修改的上游代码路径**，可作为阶段 3 的干净对照。

**[DEVIATION-7] Poisson 重建的上游 WARNING（非致命）**
- `[WARNING] FEMTree.Initialize.inl (Line 193) Found bad data: 55`
- `[WARNING] FEMTree.IsoSurface.specialized.inl (Line 1858) Extract bad average roots: 7`
- 属 Open3D 内置 PoissonRecon 的常见提示（输入点云含少量退化法向），网格正常生成且顶点无 NaN。已如实记录。

**未做的事（阶段 2 不要求，留待后续阶段）**：D1 提到的"可视化"、渲染指标（SSIM/LPIPS）、几何指标均属阶段 4，本阶段未做。

---

## 复算命令（验收方可独立重跑）

```bash
source /scratch/e1351071/zju_test/env.sh
cd /scratch/e1351071/zju_test

# B1 —— 产物 + 独立复算 test PSNR（约 1 分钟，需 GPU）
ls -l outputs/baseline/gs_truck/point_cloud/iteration_7000/point_cloud.ply
tr '\r' '\n' < logs/s21_3dgs.log | grep "ITER 7000"
python scripts/recompute_gs_psnr.py        # → test: n=32  PSNR=23.908735  L1=0.038305
head -3 outputs/metrics/gs7000_psnr_recompute.csv

# B2 —— coarse ckpt + 日志
ls -l outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt
python -c "
import torch; d=torch.load('outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt',map_location='cpu')
print(d['iteration'], {k:tuple(v.shape) for k,v in d['state_dict'].items()})"
tr '\r' '\n' < logs/s22_coarse_baseline.log | grep -nE "Starting SDF regularization|Training finished after"

# B3 —— mesh 独立核验
python scripts/verify_mesh.py outputs/runs/mesh_baseline/*.ply

# B4 / B5
cat outputs/baseline/INDEX.md
sed -n '/S2.4 三步墙钟汇总/,/产物索引/p' notes/RUNLOG.md

# 仓库未被改动
cd repo/SuGaR && git status --porcelain | wc -l && git log -1 --format=%H
```

## 日志与产物路径

| 内容 | 绝对路径 |
|---|---|
| 3DGS 训练日志 | `/scratch/e1351071/zju_test/logs/s21_3dgs.log` |
| coarse 训练日志 | `/scratch/e1351071/zju_test/logs/s22_coarse_baseline.log` |
| coarse 首次失败日志（留档） | `/scratch/e1351071/zju_test/logs/s22_coarse_baseline_FAILED_nopathsep.log` |
| mesh 提取日志 | `/scratch/e1351071/zju_test/logs/s23_mesh_baseline.log` |
| mesh 独立核验输出 | `/scratch/e1351071/zju_test/logs/verify_mesh_baseline.log` |
| 3DGS 点云 | `/scratch/e1351071/zju_test/outputs/baseline/gs_truck/point_cloud/iteration_7000/point_cloud.ply` |
| coarse 模型 | `/scratch/e1351071/zju_test/outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt` |
| mesh | `/scratch/e1351071/zju_test/outputs/runs/mesh_baseline/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply` |
| 逐视角 PSNR CSV | `/scratch/e1351071/zju_test/outputs/metrics/gs7000_psnr_recompute.csv` |
| 产物索引 | `/scratch/e1351071/zju_test/outputs/baseline/INDEX.md` |
| 运行脚本 | `/scratch/e1351071/zju_test/scripts/run_s2{1,2,3}_*.sh`、`chain_s22_to_s23.sh`、`verify_mesh.py`、`recompute_gs_psnr.py` |

---

## 收尾审计（06:15–06:16）

```
cd repo/SuGaR
git status --porcelain | wc -l   → 0        （源码零改动，仅生成 __pycache__）
git log -1 --format=%H           → 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44
git reflog                       → 仅 "clone: from https://github.com/Anttwo/SuGaR.git"
git stash list                   → 空
```
未 commit、未删除任何文件、未修改计划与预注册假设。

**[DEVIATION-8] 同目录存在另一个并行执行者（阶段 3/4）**
审计发现 06:05–06:14 出现非本任务产生的文件：`repo/SuGaR_dev/`（DNC 开发副本，3 处改动）、
`scripts/smoke_dnc.sh`、`scripts/test_dnc_normal.py`、`notes/RUNLOG_eval.md`、`notes/SELF_CHECK_阶段3_实现.md`、
`notes/SELF_CHECK_阶段4_脚本.md`、`notes/dnc_stage3.patch`、`outputs/smoke/`、`outputs/vis/`、
`outputs/metrics/{summary.csv, render_*, geometry_*, meshvis_*, test_dnc_normal.json, crosscheck/}`。
- 对方所有代码改动落在 `repo/SuGaR_dev`，**未触碰 `repo/SuGaR`**，故阶段 1/2 的基线用的是未修改的上游代码，正确性不受影响。
- 唯一影响是**计时**：对方 smoke 测试占 GPU 的时段（06:05–06:11）与我的 S2.3（06:05:19–06:11:41）重叠，
  S2.3 的 6m22s 可能偏慢，不宜当作纯净性能基准；S2.1、S2.2 期间 GPU0 独占，计时可信。
- 本执行者未修改、未删除任何他人文件。
