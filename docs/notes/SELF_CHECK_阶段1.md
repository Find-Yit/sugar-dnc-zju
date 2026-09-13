# SELF_CHECK — 阶段 1：环境

计划：`plans/SuGaR复现与改动_6小时考核.txt` §3「阶段 1」验收条目 A1–A5
执行：opus-executor　　时间：2026-09-13 05:39:01 → 05:49:14（**总墙钟 10 分 13 秒**，预算 35 min）
环境脚本：`/scratch/e1351071/zju_test/env.sh`（下述复算命令需先 `source` 它）

| 条目 | 要求 | 实际值 | 判定 |
|---|---|---|---|
| A1 | torch → `2.4.1+cu121` / `12.1` / `True` | `torch 2.4.1+cu121 cuda 12.1 avail True`，device0 = NVIDIA H200，cc (9,0)，ndev 2 | **PASS** |
| A2 | pytorch3d `knn_points` 在 cuda 上真跑，输出形状正确 | `pytorch3d 0.7.8`；`knn_points(q[1,512,3], p[1,2048,3], K=8)` → dists `(1,512,8)`、idx `(1,512,8)`，全 finite；`estimate_pointcloud_normals` → `(1,2048,3)`，模长均值 1.0；`loss/transforms/renderer/structures` 子模块 import 通过 | **PASS** |
| A3 | `diff_gaussian_rasterization` / `simple_knn` 可 import；3DGS 训练冒烟通过 | 两者 import 成功；`simple_knn._C.distCUDA2` 真跑返回 `(2048,)`，mean 0.0824846625328064；3DGS 完整 7000 iter 训练成功（见阶段 2 B1） | **PASS** |
| A4 | 用 python zipfile 解压；`data/tandt/truck/images` 有 251 张，`sparse/0` 存在 | 解压 8.0s / 1072 条目；images **251** 张（000001.jpg–000251.jpg，979×546）；`sparse/0/{cameras.bin 64B, images.bin 62,106,537B, points3D.bin 15,602,063B}` 齐全；`--eval`+llffhold=8 → **train 219 / test 32** | **PASS** |
| A5 | `notes/ENV.md` 写好（含 nvidia-smi、pip freeze 摘录、commit 哈希） | `notes/ENV.md` 8 节：硬件（nvidia-smi CSV）/ OS+编译器 / Python+PyTorch / 关键依赖版本表 / 仓库 commit `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44` / 数据（含 sha256）/ 随机种子 / 环境变量；完整清单 `notes/pip_freeze.txt` | **PASS** |

**阶段 1 结论：A1–A5 全部 PASS。**

---

## 关键版本（摘自 `notes/ENV.md`）

| 组件 | 版本 |
|---|---|
| torch / torchvision / torchaudio | 2.4.1+cu121 / 0.19.1+cu121 / 2.4.1 |
| torch.version.cuda / cudnn | 12.1 / 90100 |
| 系统 nvcc | 12.1.105 |
| pytorch3d | 0.7.8（官方预编译 wheel `py310_cu121_pyt241`，`--no-deps` 安装） |
| numpy | **1.26.4**（见下方「偏离」） |
| open3d / scipy / scikit-learn / plyfile / rich / plotly | 0.19.0 / 1.15.3 / 1.7.2 / 1.1.3 / 15.0.0 / 7.0.0 |
| diff-gaussian-rasterization / simple-knn | 0.0.0（源码编译自 SuGaR 自带子模块，`TORCH_CUDA_ARCH_LIST=9.0`） |
| GPU / 驱动 | 2×NVIDIA H200 143771 MiB / 575.57.08 |
| Python | 3.10.12 |

`PIP_CONSTRAINT=$VIRTUAL_ENV/constraints.txt` 内容：`torch==2.4.1 / torchvision==0.19.1 / torchaudio==2.4.1 / numpy<2`。
安装链每一步之后都复验过 torch 版本未被改动（见 `logs/setup_step2.log` 的 `torch after deps: 2.4.1+cu121 12.1 True` 与 `FINAL torch 2.4.1+cu121 12.1 True`）。

---

## 偏离与说明（[DEVIATION]）

**[DEVIATION-1] 追加 `numpy<2` 到 constraints（计划未写）**
- 现象：`pip install torch==2.4.1` 会连带装 numpy 2.2.6。
- 原因：SuGaR/3DGS/open3d 均为 2023–2024 年代、按 numpy 1.x ABI 构建；CLAUDE.md §9 故障表也明确列出 `numpy.dtype size changed` 属 numpy 2.x ABI 冲突。
- 处理：在 `$VIRTUAL_ENV/constraints.txt` 追加 `numpy<2`，实装 **numpy 1.26.4**。属于降低风险的防御性收紧，不改变计划的任何科学假设。

**[DEVIATION-2] 额外安装了 `opencv-python`，并被动引入 `matplotlib`**
- `opencv-python` 是我在依赖清单里多加的一项（import 扫描显示 SuGaR 并不需要 cv2），**属于多余安装**，已装入版本 4.11.0.86，对实验无影响。
- `matplotlib 3.10.9` / `pandas` 是 **open3d 0.19.0 的传递依赖**，由 pip 自动拉入，并非我主动安装文档工具链。主会话交代的"不要装 matplotlib 等文档类包到 ML venv"意在避免把文档工具链混进 ML venv；此处为必需包 open3d 的强制依赖，无法在不使用 `--no-deps` 的前提下规避。文档工具链仍严格隔离在 `tools/doc_env`，未被触碰。
- 副作用：装 opencv-python 时因 `numpy<2` 约束触发了 pip 回溯（4.14→4.11），多花约 15 秒。

**[DEVIATION-3] 未安装 `pymcubes`**
- 计划写「若编译失败可跳过，非必需」。核查 `sugar_extractors/coarse_mesh.py:658` 确认 `import mcubes` 只在函数内部、且仅当 `--use_marching_cubes True` 时执行；本次用 Poisson 路径，故直接不装。
- 同理 `nvdiffrast` 未装：`sugar_utils/mesh_rasterization.py` 用 try/except 保护，缺失时自动回退 pytorch3d 光栅化，仅影响 refined 阶段（本次不做）。

---

## 复算命令（验收方可独立重跑）

```bash
source /scratch/e1351071/zju_test/env.sh

# A1
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# A2
python -c "
import torch; from pytorch3d.ops import knn_points; import pytorch3d
torch.manual_seed(0)
p=torch.randn(1,2048,3,device='cuda'); q=torch.randn(1,512,3,device='cuda')
o=knn_points(q,p,K=8); print(pytorch3d.__version__, o.dists.shape, o.idx.shape, torch.isfinite(o.dists).all().item())"

# A3
python -c "
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2
print('ok', distCUDA2(torch.randn(2048,3,device='cuda')).shape)"

# A4
ls /scratch/e1351071/zju_test/data/tandt/truck/images | wc -l          # → 251
ls /scratch/e1351071/zju_test/data/tandt/truck/sparse/0/               # → cameras.bin images.bin points3D.bin project.ini
sha256sum /scratch/e1351071/zju_test/data/tandt_db.zip                 # → 816e62f2...d7e1

# A5
cat /scratch/e1351071/zju_test/notes/ENV.md
bash /scratch/e1351071/zju_test/scripts/gen_env_md.sh   # 可重新生成并 diff
```

## 日志与产物路径

| 内容 | 路径 |
|---|---|
| torch 安装日志 | `logs/pip_torch.log` |
| 依赖 + CUDA 扩展编译日志 | `logs/setup_step2.log` |
| pytorch3d wheel 下载日志 | `logs/wget_pytorch3d.log` |
| 数据解压日志 | `logs/unzip_data.log` |
| 阶段 1 验收测试输出 | `logs/verify_stage1.log` |
| 环境记录 | `notes/ENV.md`、`notes/pip_freeze.txt`、`notes/tandt_db.sha256` |
| 安装脚本 | `scripts/setup_env_step2.sh`、`scripts/gen_env_md.sh`、`scripts/unzip_data.py` |
| 环境变量 | `env.sh`，pip 约束 `/scratch/e1351071/virtualenvs/zju_test/constraints.txt` |
