# scripts_a100 —— A100 侧提取端实验的脚本与复现命令

本目录是 A100 机器上实际跑实验用的脚本副本（原位置 `$PROJ_ROOT/scripts/m/` 与
`repo/sugar-dnc-zju/scripts/eval/`）。所有脚本都假定先 `source $PROJ_ROOT/env.sh`，
且在 NGC Singularity 容器内运行：

```bash
singularity exec --nv --bind /scratch:/scratch \
  /app/apps/containers/pytorch/pytorch_23.05_py3.sif bash
source /scratch/users/nus/e1351071/test_zju/env.sh
```

## 文件清单

| 文件 | 作用 |
|---|---|
| `extract_one.sh` | 单组 mesh 提取（调 `extract_mesh.py`），带 SIGSEGV 重试与墙钟记录 |
| `eval_one.sh` | 单组几何评测（调 `eval/eval_geometry.py`），`SKIP_VIS=0` 时额外 跑 mesh 可视化 |
| `run_grid.sh` | 按 `grid_*.txt` 批量跑扫参网格（多 GPU 分派） |
| `grid_status.sh` | 查看网格进度 |
| `grid_*.txt` | 扫参网格定义，每行 `TAG\|GPU\|COARSE_PT\|EXTRA` |
| `summarize.py` | 把各 run 的 `geometry_*.json` 汇总成 `outputs/metrics/summary_m.csv` |
| `meshvis_summary.py` | 把 `meshvis_*.json` 汇总成 `outputs/metrics/meshvis_summary_m.csv`（覆盖率表） |
| `make_figs_meshvis.py` | 生成 `outputs/figures/m_nscc/` 的四张对比图 |
| `make_figs_a100.py` | 生成本次交付的三张新图（碎片对比 / 天空封口证据 / 扫参图） |
| `render_fragments_views.py` | 碎片分量着色渲染（open3d 聚类 + pytorch3d 光栅化） |
| `verify_independent.py` | 独立复算校验（从 .ply 重新算拓扑指标，验证 summary 数字） |
| `eval/render_mesh_views.py` | mesh 法向图 / 深度图渲染 + **覆盖率指标**（定义见 `COVERAGE_METRIC.md`） |
| `eval/eval_geometry.py` | 几何指标 G1–G5 |
| `eval/_common.py` | 固定测试视角常量、PNG 保存、turbo 色表 |

## 复现流程（剪枝 → 网格 → 评测 → 汇总 → 复算 → 可视化）

### 0. 前置产物
`outputs/baseline/gs_truck/`（3DGS 7k）与 `outputs/runs/coarse_base_seed0/sugarcoarse_*/15000.pt`（SuGaR coarse 15k）。

### 1. 剪枝（M2-B，仅剪枝组需要）

```bash
cd $PROJ_ROOT/repo/sugar-dnc-zju
python prune_coarse_model.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 \
  -m $PROJ_ROOT/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -o $PROJ_ROOT/outputs/m2/pruned_p95 \
  --rule keep_large --knn_percentile 95 --min_samples 6 \
  --separate_fg_bg True --seed 0 --gpu 0
# 产出 15000_pruned.pt + prune_stats.json
```

### 2. 网格提取

```bash
# 原版基线（全默认参数 = 原版行为）
TAG=base_d10_q01 GPU=0 bash extract_one.sh

# q0（关掉分位清洗）
TAG=base_d10_q0 GPU=0 EXTRA="--vertices_density_quantile 0 --extract_seed 0" bash extract_one.sh

# M1+a（背景 Poisson 深度 9）
TAG=m1a_q0 GPU=0 EXTRA="--poisson_depth_bg 9 --vertices_density_quantile 0 --extract_seed 0" bash extract_one.sh

# 最优组合 = q0 + M2-B(p95) + M1+a(bg9)
TAG=m2p95_bg9_q0 GPU=1 \
  COARSE_PT=$PROJ_ROOT/outputs/m2/pruned_p95/15000_pruned.pt \
  EXTRA="--poisson_depth 10 --poisson_depth_bg 9 --vertices_density_quantile 0 --extract_seed 0" \
  bash extract_one.sh

# 批量扫参
bash run_grid.sh grid_wave2.txt      # 进度：bash grid_status.sh
```

### 3. 评测（几何 G1–G5）

```bash
TAG=base_d10_q01 GPU=0 bash eval_one.sh          # 只算几何
SKIP_VIS=0 TAG=base_d10_q01 GPU=0 bash eval_one.sh   # 同时出法向/深度图 + 覆盖率
# → outputs/metrics/geometry_m_<TAG>.json, meshvis_m_<TAG>.json, outputs/vis/m_<TAG>/
```

### 4. 汇总

```bash
python summarize.py        # → outputs/metrics/summary_m.csv
python meshvis_summary.py  # → outputs/metrics/meshvis_summary_m.csv（覆盖率表）
```

### 5. 独立复算（校验汇总数字没写错）

```bash
python verify_independent.py   # 从 .ply 重新算拓扑，与 summary_m.csv 逐项比对
# → outputs/metrics/verify_independent.json
```

### 6. 可视化

```bash
# 法向图 / 深度图（4 个固定视角）
python eval/render_mesh_views.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --mesh_path $PROJ_ROOT/outputs/m/mesh_base_d10_q01/sugarmesh_*.ply \
  --run_name m_base_d10_q01 --gpu 0

# 碎片分量着色图（红=<100面碎片，灰=最大分量，浅色=其余）
python render_fragments_views.py \
  --scene_path $PROJ_ROOT/data/tandt/truck \
  --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
  --mesh_path $PROJ_ROOT/outputs/m/mesh_base_d10_q01/sugarmesh_*.ply \
  --tag base_d10_q01 --out_dir <figures_dir> --gpu 0

# 拼合成交付图
python make_figs_meshvis.py    # → outputs/figures/m_nscc/*.png
python make_figs_a100.py       # → results_a100/figures/fig_*.png（300 dpi，中文）
```

> 中文字体：所有绘图脚本通过 `assets/fonts/NotoSansCJKsc-Regular.otf` 注册，
> 不要依赖系统字体，否则中文变方框。
