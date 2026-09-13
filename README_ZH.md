# SuGaR + 深度-法向一致性正则（DNC）

在 [SuGaR](https://github.com/Anttwo/SuGaR)（CVPR 2024）的 **coarse 训练主循环**里加入一项
**深度-法向一致性正则**（Depth-Normal Consistency，下称 DNC），并在 Tanks & Temples 的 Truck
场景上做 λ ∈ {0, 0.05, 0.2} 三组完整对照（训练 + 泊松网格提取 + 统一评测）。

思路来源是 [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting)（SIGGRAPH 2024）
提出的 depth-normal consistency 损失；本仓库把它移植到 SuGaR 的三维椭球表示上并自行实现，
**没有复制 2DGS 的任何代码**。

> 本仓库基于 SuGaR 官方仓库 commit `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`（2024-09-24）。
> 全部改动集中在下面三个文件，git 历史的起点就是官方 commit，便于逐行对照 diff。

---

## 1. 改动内容

| 文件 | 改动 |
|---|---|
| `sugar_utils/dnc_utils.py` | **新增**。自包含的工具模块：NDC 内参提取、深度反投影、结构化点图求法向、有效像素掩码、L_DNC 损失、中间量可视化。只依赖 torch / numpy / PIL，可脱离 SuGaR 单独做单元测试。 |
| `sugar_trainers/coarse_density.py` | 在训练循环中渲染深度图 D 与法向图 N，计算 L_DNC 并加进总损失；逐迭代日志与中间量落盘。 |
| `train_coarse_density.py` | 新增命令行参数 `--dnc_factor`、`--dnc_start`、`--seed` 等。 |

### 损失定义

设 `D` 为渲染深度图（相机系 z，背景填 `max_depth`），`N` 为逐像素 α 合成后的高斯法向
（取最短 scale 轴，逐高斯翻转为朝向相机），则

```
P(u,v) = unproject( x_ndc(u), y_ndc(v), D(u,v) )            # 反投影成相机系点图
N_d    = normalize( ∂P/∂u × ∂P/∂v )                          # 中心差分 + 叉乘
M      = { (u,v) : D < 0.98·D_max  ∧  |∇D| < 0.05·D  ∧  ‖N_raw‖ > 0.1  ∧  非边缘 2 像素 }
L_DNC  = mean_{(u,v)∈M} ( 1 − |cos( N(u,v), N_d(u,v) )| )
L      = L_SuGaR + λ · L_DNC
```

取绝对值是为了消除三维椭球最短轴的法向**符号歧义**（2DGS 的二维圆盘没有这个问题）。
掩码里的 `‖N_raw‖ > 0.1` 是数值保护：α 合成后低不透明度像素的法向接近零向量、方向无意义。

`λ = 0`（默认）时**完全不进入 DNC 分支**（不额外渲染 D 与 N），保证基线的代码路径与原版一致。
DNC 默认从第 9000 迭代起生效，与 SuGaR 自身的 SDF 正则同期。

### 可检验假设（实验前预注册）

- **H1（几何）**：λ>0 时网格漂浮碎片分量数下降、稀疏点到网格距离中位数下降、法向更平滑。
- **H2（渲染代价）**：测试视角 PSNR 变化在 ±0.5 dB 内；超出则如实报告为代价。
- **H3（叠加关系）**：DNC 与 SuGaR 原有的 SDF 法向损失不冲突，训练中两者同时下降。

结论与全部数字见 `report.pdf` 与 `results/summary.csv`。

---

## 2. 环境安装

实测环境：Ubuntu 22.04 / glibc 2.35、NVIDIA H200（sm_90，驱动 575.57.08）、系统 nvcc 12.1、Python 3.10.12。

> 选 **torch 2.4.1 + cu121** 的唯一理由是：pytorch3d 0.7.8 有 `py310_cu121_pyt241` 的官方预编译 wheel，
> 而 pytorch3d 源码编译要 20–40 分钟。若换 torch 版本，必须先确认 pytorch3d 有对应 wheel。

```bash
python3.10 -m venv .venv && source .venv/bin/activate

# 1) torch —— 必须显式 pin，否则 pip 会装到 CUDA 13 的新版本
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --retries 10 --timeout 60
printf 'torch==2.4.1\ntorchvision==0.19.1\ntorchaudio==2.4.1\nnumpy<2\n' > .venv/constraints.txt
export PIP_CONSTRAINT=$PWD/.venv/constraints.txt      # 之后所有 pip 都受此约束，防止被偷偷升级

# 2) numpy < 2（SuGaR / 3DGS / open3d 生态按 numpy 1.x 构建）
pip install "numpy<2" ninja

# 3) pytorch3d 官方预编译 wheel（--no-deps，防止它拖动 torch）
wget -c https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
pip install --no-deps pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
pip install fvcore iopath

# 4) 其余依赖
pip install open3d plyfile rich tqdm plotly scipy scikit-learn opencv-python

# 5) 3DGS 的两个 CUDA 扩展 —— 必须用仓库自带的子模块目录，不要装 PyPI 上的同名包
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST="9.0"                     # H200 = sm_90，按自己的卡改
cd gaussian_splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation --no-deps . && cd -
cd gaussian_splatting/submodules/simple-knn            && pip install --no-build-isolation --no-deps . && cd -

# 6) 复验
python -c "import torch, pytorch3d, open3d, diff_gaussian_rasterization, simple_knn; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

---

## 3. 数据获取

使用 3D Gaussian Splatting 官方打包的 `tandt_db.zip`（含 Tanks & Temples 与 Deep Blending，
已带 COLMAP 稀疏重建）：

```bash
mkdir -p data && cd data
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
python -c "import zipfile; zipfile.ZipFile('tandt_db.zip').extractall('.')"
cd ..
# 得到 data/tandt/truck/{images, sparse/0}，251 张图像，分辨率 979×546
```

划分遵循 3DGS 的 `--eval`（llffhold = 8，按文件名排序后索引 % 8 == 0 为测试）
→ **训练 219 张 / 测试 32 张**。三组对照与网格提取全程使用同一划分。

---

## 4. 复现步骤

一条命令跑完全流程（推荐，见 `scripts/reproduce_all.sh`）：

```bash
export PROJ_ROOT=/path/to/workdir       # 产物根目录
export SUGAR_DIR=$PWD                   # 本仓库
export SCENE=$PROJ_ROOT/data/tandt/truck
bash scripts/reproduce_all.sh           # 单卡顺序执行；加 GPUS=0,1 可两卡并行
```

或者逐步执行：

### 第一步：3DGS 训练 7000 迭代（三组共用同一检查点，保证公平）

```bash
python gaussian_splatting/train.py \
  -s $SCENE -m $PROJ_ROOT/outputs/baseline/gs_truck \
  --iterations 7000 --save_iterations 7000 --test_iterations 7000 --eval
```

### 第二步：coarse SuGaR 训练（三组，唯一差别是 `--dnc_factor`）

> ⚠️ `-c` 的路径**必须以 `/` 结尾**：SuGaR 用字符串拼接 `gs_output_path + 'cameras.json'`，
> 这是上游的既定用法（官方 `train_full_pipeline.py` 也会自动补斜杠），不是 bug。

```bash
GS=$PROJ_ROOT/outputs/baseline/gs_truck/     # 注意结尾的斜杠

# λ = 0（基线；代码路径与原版 SuGaR 一致）
python train_coarse_density.py -s $SCENE -c $GS -i 7000 \
  -o $PROJ_ROOT/outputs/runs/coarse_base_seed0 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0    --dnc_start 9000

# λ = 0.05
python train_coarse_density.py -s $SCENE -c $GS -i 7000 \
  -o $PROJ_ROOT/outputs/runs/coarse_dnc005 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0.05 --dnc_start 9000

# λ = 0.2
python train_coarse_density.py -s $SCENE -c $GS -i 7000 \
  -o $PROJ_ROOT/outputs/runs/coarse_dnc02 \
  --eval True --gpu 0 --seed 0 --dnc_factor 0.2  --dnc_start 9000
```

训练默认从 3DGS 的 7000 迭代续训到 15000：7000–9000 熵正则，9000 做一次 `opacity < 0.5`
的硬剪枝，9000 起进入 SDF 正则（DNC 也在此时生效）。
λ>0 时会额外产出 `dnc_log.csv`（每 100 迭代一行）与 `dnc_vis/`（每 1000 迭代的 D / N / N_d / 掩码）。

### 第三步：泊松网格提取（三组参数完全一致）

```bash
for RUN in coarse_base_seed0 coarse_dnc005 coarse_dnc02; do
  python extract_mesh.py -s $SCENE -c $GS -i 7000 \
    -m $PROJ_ROOT/outputs/runs/$RUN/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
    -l 0.3 -d 200000 --eval True --gpu 0 \
    -o $PROJ_ROOT/outputs/runs/${RUN/coarse_/mesh_}
done
```

> `-d 200000` 的语义是**前景网格与背景网格各自抽取到 20 万个三角形**，
> 合并后总面数约 37 万，属正常现象，不是“输出 20 万面”。

### 第四步：评测与汇总

```bash
for RUN in coarse_base_seed0 coarse_dnc005 coarse_dnc02; do
  bash scripts/eval/run_eval_for_run.sh $RUN \
    $PROJ_ROOT/outputs/runs/$RUN/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt \
    $PROJ_ROOT/outputs/runs/${RUN/coarse_/mesh_}/sugarmesh_*.ply 0
done
python scripts/eval/summarize.py          # -> outputs/metrics/summary.csv
```

### 第五步：图表与报告

```bash
tools/doc_env/bin/python scripts/deliverables/make_figures.py   # -> outputs/figures/fig*.png|pdf
tools/doc_env/bin/python scripts/deliverables/make_report.py    # -> outputs/reports/report.docx
```

---

## 5. 评测指标

全部由 `scripts/eval/` 下的脚本统一计算，三组使用同一套代码与参数。

**渲染**（32 个测试视角，用各自 coarse SuGaR 15000 迭代的高斯渲染）
| 指标 | 定义 | 方向 |
|---|---|---|
| PSNR | 逐图 MSE 取 −10·log10 后求均值，dB | ↑ |
| SSIM | 3DGS 官方实现的 11×11 高斯窗 | ↑ |
| LPIPS-VGG | VGG 特征上的感知距离 | ↓ |

**几何**（Truck 无真值网格，以 COLMAP 稀疏点作独立参考；只取 track 长度 ≥ 3 且落在前景包围盒内的点）
| 指标 | 定义 | 方向 |
|---|---|---|
| G1 | 稀疏点到网格表面的无符号距离（Open3D `RaycastingScene`）的 mean / median / P90，同时给出除以相机空间尺度的相对值 | ↓ |
| G2 | G1 距离小于相机空间尺度 0.5% / 1% 的点占比 | ↑ |
| G3 | 网格顶点到最近稀疏点的平均距离（Chamfer 另一半，仅统计前景顶点）；稀疏点本身稀疏，只作参考 | ↓ |
| G4 | 顶点数、面数、连通分量数、最大分量面数占比、面数 < 100 的碎片分量数、非流形边、边界边 | 碎片 ↓ / 最大分量占比 ↑ |
| G5 | 相邻面二面角的 mean / median / P90（度） | ↓（过低可能是过度平滑） |

**训练代价**：墙钟、每迭代耗时、峰值显存、结束时高斯数。

---

## 6. 可复现性说明

- 3DGS 的 `safe_state()` 固定 `random` / `numpy` / `torch` 的种子为 0；coarse 训练显式传 `--seed 0`。
- 但 **CUDA 光栅化（diff-gaussian-rasterization）的原子累加本身非确定**，即使固定种子，
  逐次运行也会有微小数值差异。本项目把 0.01 dB 量级的 PSNR 差异视为噪声，不作为结论依据。
- 本仓库不包含数据集与权重（见 `.gitignore`）；`results/` 下只放体积很小的指标 CSV / JSON 与图表。

---

## 7. 引用

- Guédon A., Lepetit V. *SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh
  Reconstruction and High-Quality Mesh Rendering.* CVPR 2024.
  <https://github.com/Anttwo/SuGaR>
- Kerbl B., Kopanas G., Leimkühler T., Drettakis G. *3D Gaussian Splatting for Real-Time
  Radiance Field Rendering.* ACM TOG (SIGGRAPH) 2023.
  <https://github.com/graphdeco-inria/gaussian-splatting>
- Huang B., Yu Z., Chen A., Geiger A., Gao S. *2D Gaussian Splatting for Geometrically Accurate
  Radiance Fields.* SIGGRAPH 2024. —— **本项目 DNC 损失的思路来源**
  <https://github.com/hbb1/2d-gaussian-splatting>
- Knapitsch A., Park J., Zhou Q.-Y., Koltun V. *Tanks and Temples: Benchmarking Large-Scale
  Scene Reconstruction.* ACM TOG 2017. —— Truck 场景来源
- Kazhdan M., Hoppe H. *Screened Poisson Surface Reconstruction.* ACM TOG 2013. —— 网格提取算法

许可证沿用上游 SuGaR 仓库的许可证。
