# RUNLOG — SuGaR 复现与改动（6 小时考核）

执行者：opus-executor（claude-opus-5, effort max）
计划：`/scratch/e1351071/zju_test/plans/SuGaR复现与改动_6小时考核.txt`
项目根：`PROJ_ROOT=/scratch/e1351071/zju_test`
ML venv：`/scratch/e1351071/virtualenvs/zju_test`
环境脚本：`/scratch/e1351071/zju_test/env.sh`（每个 shell 先 `source`）

---

## 阶段 1：环境

### 1.0 起始环境自检 — 2026-09-13 05:39:01 Asia
命令：
```
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
/scratch/e1351071/virtualenvs/zju_test/bin/python -c "import sys;print(sys.version)"
/scratch/e1351071/virtualenvs/zju_test/bin/pip list
/usr/local/cuda/bin/nvcc --version
```
结果：
- GPU0/GPU1 = NVIDIA H200，143771 MiB，0 MiB 已用，driver 575.57.08
- Python 3.10.12 (GCC 11.4.0)
- venv 干净：仅 pip 22.0.2 + setuptools 59.6.0
- nvcc release 12.1, V12.1.105
- /scratch 可用 3.2P
- repo/SuGaR commit `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`（2024-09-24 "Minor fix."）
- `gaussian_splatting/submodules/{diff-gaussian-rasterization,simple-knn}` 存在；glm 第三方目录存在

### 1.1 写 env.sh + pip constraints — 05:39:32
- `$PROJ_ROOT/env.sh`（含 HF_HOME/TORCH_HOME 隔离、`PIP_CONSTRAINT`、`TORCH_CUDA_ARCH_LIST=9.0`）
- `$VIRTUAL_ENV/constraints.txt` = torch==2.4.1 / torchvision==0.19.1 / torchaudio==2.4.1

### 1.2 后台启动三条长任务 — 05:39:32 / 05:39:38
| 任务 | 命令 | 日志 |
|---|---|---|
| torch 安装 | `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --retries 10 --timeout 60` | `logs/pip_torch.log` |
| pytorch3d wheel 下载 | `wget -c https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl -P wheels/` | `logs/wget_pytorch3d.log` |
| 数据解压 | `/usr/bin/python3.10 scripts/unzip_data.py`（python zipfile，本机无 unzip） | `logs/unzip_data.log` |

结果：
- wget 05:39:33 完成，20,521,514 字节 → `wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl`
- 解压 05:39:46 完成（8.0s，1072 条目）→ `data/tandt/{truck,train}`、`data/db/{drjohnson,playroom}`
- `data/tandt/truck`：images 251 张（979x546），`sparse/0/{cameras.bin,images.bin,points3D.bin}` 齐全，180M

### 1.3 torch 安装 — 05:39:32 → 05:42:46（墙钟 3m14s）
命令（后台）：`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --retries 10 --timeout 60`
日志：`logs/pip_torch.log`
结果：`torch-2.4.1 torchvision-0.19.1 torchaudio-2.4.1`，依赖含 `nvidia-cuda-runtime-cu12-12.1.105`（确认 cu121 构建）；
      附带装入 `numpy-2.2.6`（随后在 1.4 降级为 <2）。
复验：`torch 2.4.1+cu121 cuda 12.1 avail True abi False`  → **A1 PASS**

### 1.4 后续依赖 + CUDA 扩展编译 — 05:42:48 起
脚本：`scripts/setup_env_step2.sh`（后台链，等 torch pip 退出后自动执行），日志 `logs/setup_step2.log`
步骤：
1. 把 `numpy<2` 加进 `$VIRTUAL_ENV/constraints.txt`（2024 年代 SuGaR/3DGS/open3d 生态按 numpy 1.x 构建），`pip install "numpy<2" ninja`
2. `pip install --no-deps wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl` + `pip install fvcore iopath`
3. `pip install open3d plyfile rich tqdm plotly scipy scikit-learn opencv-python`
4. `cd gaussian_splatting/submodules/diff-gaussian-rasterization && TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation --no-deps .`
5. `cd gaussian_splatting/submodules/simple-knn && TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation --no-deps .`
每步后复验 torch 版本未被改动。

### 1.5 阶段 1 验收测试 — 05:48:26 → 05:48:51
命令：`source env.sh && python - <<PY ... PY`（内联脚本，日志 `logs/verify_stage1.log`）
输出：
```
A1 torch: 2.4.1+cu121 | cuda: 12.1 | avail: True | ndev: 2
A1 device0: NVIDIA H200 cc: (9, 0)
A2 pytorch3d: 0.7.8 | knn dists (1, 512, 8) | idx (1, 512, 8) | finite: True
A2 estimate_pointcloud_normals: (1, 2048, 3) | norm~1: 1.0
A2 pytorch3d submodules import OK (loss/transforms/renderer/structures)
A3 diff_gaussian_rasterization import OK; simple_knn distCUDA2: (2048,) mean 0.0824846625328064
A3 open3d: 0.19.0 | numpy: 1.26.4 | scipy: 1.15.3 | sklearn: 1.7.2
ALL_STAGE1_IMPORT_CHECKS_PASS
```
CUDA 扩展编译墙钟：`diff-gaussian-rasterization` + `simple-knn` 合计 05:45:5x → 05:48:26（约 2m30s）。

### 1.6 生成 ENV.md — 05:49:14
命令：`bash scripts/gen_env_md.sh`（自动抓取实时环境，避免手抄）
产物：`notes/ENV.md`、`notes/pip_freeze.txt`、`notes/tandt_db.sha256`
数据 sha256：`816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`

**阶段 1 总墙钟：05:39:01 → 05:49:14 ≈ 10 分 13 秒**（预算 35 min）

---

## 阶段 2：基线流水线

### S2.1 3DGS 7000 iter（GPU0）— 05:48:58 → 05:51:11，墙钟 **2m13s**
脚本：`scripts/run_s21_3dgs.sh`，日志 `logs/s21_3dgs.log`
命令：
```
cd $PROJ_ROOT/repo/SuGaR
CUDA_VISIBLE_DEVICES=0 python gaussian_splatting/train.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -m $PROJ_ROOT/outputs/baseline/gs_truck \
  --iterations 7000 --save_iterations 7000 --test_iterations 7000 --eval
```
关键输出：
```
Number of points at initialisation :  136029
[ITER 7000] Evaluating test: L1 0.03830549860140309 PSNR 23.908734679222107
[ITER 7000] Evaluating train: L1 0.03396476022899151 PSNR 24.76439895629883
[ITER 7000] Saving Gaussians
Training complete.
```
产物：`outputs/baseline/gs_truck/point_cloud/iteration_7000/point_cloud.ply`（460,021,692 字节）、
`cameras.json`（251 个相机）、`cfg_args`（`eval=True` 已确认）、`input.ply`
训练速度：约 85 it/s（H200）

### S2.2 第一次尝试失败（05:51:25 → 05:51:33）—— 路径尾斜杠问题
报错：
```
FileNotFoundError: .../outputs/baseline/gs_truckcameras.json
  sugar_scene/cameras.py:34  with open(gs_output_path + 'cameras.json')
```
根因：SuGaR 用**字符串拼接**而非 `os.path.join` 拼 `cameras.json`，`-c` 必须以 `/` 结尾。
上游官方 `train_full_pipeline.py:130-131` 也显式做 `if gs_checkpoint_dir[-1] != os.path.sep: gs_checkpoint_dir += os.path.sep`，
即**尾斜杠是上游约定的用法**，不是代码 bug。处理：只改自己的运行脚本命令行（`-c .../gs_truck/`），**未改动任何仓库代码**。
失败日志留档：`logs/s22_coarse_baseline_FAILED_nopathsep.log`（未删除）。

### S2.2 coarse SuGaR baseline（GPU0）— 05:52:34 起
脚本：`scripts/run_s22_coarse_baseline.sh`，日志 `logs/s22_coarse_baseline.log`
命令：
```
cd $PROJ_ROOT/repo/SuGaR
python train_coarse_density.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -c $PROJ_ROOT/outputs/baseline/gs_truck/ \
  -i 7000 \
  -o $PROJ_ROOT/outputs/runs/coarse_baseline \
  --eval True --gpu 0
```
（按主会话指示，本阶段**不传** `--dnc_factor`）
05:53:12 状态：已进入训练循环，`Iteration: 7000`，`Starting entropy regularization.`，GPU0 显存 7.7 GB / 利用率 100%。

S2.2 完成：05:52:34 → 06:05:18，墙钟 **12m44s**。
关键日志：
```
Pruning gaussians with low-opacity for further optimization...
Pruning finished: 429075 gaussians left.          (iter 9000 硬剪枝)
---INFO--- Starting SDF regularization.           (log 行 288)
---INFO--- Starting SDF estimation loss.
---INFO--- Starting SDF better normal loss.
Iteration: 15000
loss: 0.091825  [15000/15000] computed in 0.3697449564933777 minutes.
Number of gaussians used for sampling in SDF regularization: tensor(140831, device='cuda:0')
Training finished after 15000 iterations with loss=0.0918252244591713.
Final model saved.
```
产物：`outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`
训练速度：熵正则阶段（7000–9000）约 0.05 min/200 iter；SDF 正则阶段（9000–15000）约 0.37 min/200 iter。
GPU0 峰值显存约 10.1 GB，利用率 96–100%。

### S2.3 mesh 提取（GPU0）— 06:05:19 起
脚本：`scripts/run_s23_mesh_baseline.sh`（由 `scripts/chain_s22_to_s23.sh` 在 S2.2 结束后自动触发），日志 `logs/s23_mesh_baseline.log`
命令：
```
cd $PROJ_ROOT/repo/SuGaR
python extract_mesh.py \
  -s $PROJ_ROOT/data/tandt/truck \
  -c $PROJ_ROOT/outputs/baseline/gs_truck/ \
  -i 7000 \
  -m $PROJ_ROOT/outputs/runs/coarse_baseline/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
  -l 0.3 -d 200000 \
  --eval True --gpu 0 \
  -o $PROJ_ROOT/outputs/runs/mesh_baseline
```
参数确认（日志头部）：Surface levels [0.3]、Decimation targets [200000]、Project mesh on surface points True、
Use eval split True、Use marching cubes False、Use vanilla 3DGS False。

S2.3 完成：06:05:19 → 06:11:41，墙钟 **6m22s**。
关键日志：
```
Foreground mesh: TriangleMesh with 1664964 points and 3271742 triangles.
Background mesh: TriangleMesh with 2289399 points and 4500788 triangles.
Processing decimation target: 200000   （前景/背景各自 decimate 后再 merge）
Merging foreground and background meshes.
Projecting mesh on surface points to recover better details...
Mesh saved at .../sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000.ply
```
Open3D Poisson 过程中出现两条上游 WARNING（`Found bad data: 55`、`bad average roots: 7`），属 PoissonRecon 常见提示，非致命，网格正常生成。

独立核验（`scripts/verify_mesh.py`，open3d 直接读取，日志 `logs/verify_mesh_baseline.log`）：
```
n_vertices 205647 | n_triangles 370438 | file 15,304,017 B
vertices_finite True | any_nan_vertices False | vertex_colors True | vertex_normals True
edge_manifold True | vertex_manifold False
n_connected_components 2237 | largest_component_tri 157415 (42.49%) | n_components_lt100_tri 2147
bbox [-23.487,-15.537,-23.317] ~ [23.602,6.339,23.712]
```

### S2.4 三步墙钟汇总（验收条目 B5）
| 步骤 | 起 | 止 | 墙钟 |
|---|---|---|---|
| S2.1 3DGS 7000 iter | 05:48:58 | 05:51:11 | 2m13s |
| S2.2 coarse SuGaR 15000 iter | 05:52:34 | 06:05:18 | 12m44s |
| S2.3 mesh 提取 | 06:05:19 | 06:11:41 | 6m22s |
| **合计（含 05:51:25–05:51:33 失败尝试与重启间隙）** | 05:48:58 | 06:11:41 | **22m43s**（三步纯计算 21m19s） |

产物索引：`outputs/baseline/INDEX.md`

### S2.5 独立复算 3DGS test PSNR（B1 的交叉验证）— 06:14
脚本：`scripts/recompute_gs_psnr.py`（新增，只 import `gaussian_splatting` 官方模块，**不改仓库任何文件**）
```
source env.sh && python scripts/recompute_gs_psnr.py
→ test:  n=32   PSNR=23.908735   L1=0.038305      （训练日志 23.908734679222107 / 0.03830549860140309，逐位一致）
→ train: n=219  PSNR=24.800641   L1=0.034118
→ n_gaussians: 1854920
逐视角 CSV: outputs/metrics/gs7000_psnr_recompute.csv
```
说明：训练日志中的 train PSNR（24.764399）只在 **5 个**采样相机上算（`gaussian_splatting/train.py:167` 的 `range(5,30,5)`），
且训练时 `Scene(shuffle=True)` 打乱过训练相机顺序，因此与"全部 219 视角"的复算值不可能一致；
test 分支用的是**全部 32 个** test 相机，故可逐位复现，作为 B1 的独立证据。

### S2.6 收尾审计 — 06:15
```
cd repo/SuGaR && git status --porcelain | wc -l   → 0
git log -1 --format=%H                            → 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44
git reflog                                        → 只有 "clone: from https://github.com/Anttwo/SuGaR.git"
git stash list                                    → 空
```
**repo/SuGaR 源码零改动、无 commit、无 stash、无文件删除。** 仅生成了 `__pycache__`（运行副产物）。

### [并发情况说明] 同目录下存在另一个并行执行者
审计时发现本目录下有**不是本次任务产生**的文件（时间戳 06:05–06:14）：
- `repo/SuGaR_dev/`（SuGaR 的开发副本，3 处改动：`M sugar_trainers/coarse_density.py`、`A sugar_utils/dnc_utils.py`、`M train_coarse_density.py`）
- `scripts/smoke_dnc.sh`、`scripts/test_dnc_normal.py`
- `notes/RUNLOG_eval.md`、`notes/SELF_CHECK_阶段3_实现.md`、`notes/SELF_CHECK_阶段4_脚本.md`、`notes/dnc_stage3.patch`
- `outputs/smoke/`、`outputs/vis/`、`outputs/metrics/{summary.csv, render_*, geometry_*, meshvis_*, test_dnc_normal.json, crosscheck/}`

即：**有另一个执行者在并行推进阶段 3/4**，其代码改动全部落在 `repo/SuGaR_dev`，未触碰 `repo/SuGaR`。
对本次阶段 1/2 的影响评估：
1. **正确性无影响** —— `repo/SuGaR` 的 `git status` 为空、reflog 只有 clone，基线三步用的是**未修改的上游代码**；
   我的三个产物 mtime（05:51:10 / 06:05:16 / 06:11:39）与各自运行区间吻合，之后未被改写。
2. **时间可能受影响** —— 对方的 smoke 测试在 06:05–06:11 占用 GPU，与我的 S2.3 mesh 提取（06:05:19–06:11:41）重叠，
   故 S2.3 的 6m22s 可能**偏慢**，不宜作为纯净的性能基准。S2.1（05:48:58–05:51:11）与 S2.2（05:52:34–06:05:18）
   的绝大部分时段 GPU1 为 0 MiB、GPU0 独占，计时可信。
3. 本执行者**未修改、未删除**上述任何他人文件。
