# ENV.md — 运行环境记录

> ## 🔴 本机（NSCC）必须在 Singularity 容器内运行
> 裸机上没有本项目的 python/torch 生态（venv 是用容器内的 py3.10.6 建的），
> 直接 `source env.sh` 会失败。每个新 shell 先进容器：
> ```bash
> singularity exec --nv --bind /scratch:/scratch \
>   /app/apps/containers/pytorch/pytorch_23.05_py3.sif bash
> source /scratch/users/nus/e1351071/test_zju/env.sh
> ```
> 校验是否在容器内：`echo $SINGULARITY_NAME` 应输出 `pytorch_23.05_py3.sif`。

生成时间：`2026-09-13 08:58:37 Asia`  |  主机：`x1000c2s3b0n0`  |  PBS 作业：`19739513.pbs101`  |  队列：`g3`

容器：`pytorch_23.05_py3.sif`  (`/app/apps/containers/pytorch/pytorch_23.05_py3.sif`)

## 1. 硬件

```
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06, 8.0
1, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06, 8.0
2, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06, 8.0
3, NVIDIA A100-SXM4-40GB, 40960 MiB, 570.124.06, 8.0

CPU 核数: 64
内存: 503 GB
/scratch 可用: 3.1P
```

## 2. 操作系统与编译器

```
PRETTY_NAME="Ubuntu 22.04.2 LTS"
kernel: 4.18.0-553.125.1.el8_10.x86_64
glibc: ldd (Ubuntu GLIBC 2.35-0ubuntu3.1) 2.35
gcc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

## 3. Python / PyTorch

```
python: 3.10.6 (main, Mar 10 2023, 10:55:28) [GCC 11.3.0]
venv:   /scratch/users/nus/e1351071/virtualenvs/zju_test
torch:       2.4.1+cu121
torchvision: 0.19.1+cu121
torch.version.cuda: 12.1
cudnn:       90100
cuda available: True | device count: 4
device 0:    NVIDIA A100-SXM4-40GB
GLIBCXX_USE_CXX11_ABI: False
```

## 4. 关键依赖版本

```
diff_gaussian_rasterization 0.0.0
fvcore                      0.1.5.post20221221
iopath                      0.1.10
matplotlib                  3.10.9
ninja                       1.13.2
numpy                       1.26.4
open3d                      0.19.0
opencv-python               4.11.0.86
pandas                      2.3.3
pillow                      12.3.0
plotly                      7.0.0
plyfile                     1.1.3
pytorch3d                   0.7.8
rich                        15.0.0
scikit-learn                1.7.2
scipy                       1.15.3
simple_knn                  0.0.0
torch                       2.4.1
torchaudio                  2.4.1
torchvision                 0.19.1
tqdm                        4.70.1
zstandard                   0.25.0
```

完整冻结清单：`notes/pip_freeze_nscc.txt`（旧机基线：`notes/pip_freeze.txt`）

## 5. 代码仓库

```
SuGaR repo: /scratch/users/nus/e1351071/test_zju/repo/SuGaR  （baseline，应与 upstream 零 diff）
commit: 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44
date:   2024-09-24 12:14:40 -0400
subject: Minor fix.
upstream: origin	https://github.com/Anttwo/SuGaR.git (fetch)
工作区改动: 0 个文件（阶段1/2 应为 0）
```

```
SuGaR_dev repo: /scratch/users/nus/e1351071/test_zju/repo/SuGaR_dev  （阶段3 魔改 DnC）
base commit: 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44
git diff --stat:
 sugar_trainers/coarse_density.py | 226 ++++++++++++++++++++++++++++++++++++++-
 train_coarse_density.py          |  31 ++++++
 2 files changed, 256 insertions(+), 1 deletion(-)
```

## 6. 数据

```
来源: data/tandt_db.zip (Tanks&Temples + DeepBlending，3DGS 官方发布包)
大小: 682628995 字节
sha256: 816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1
使用场景: data/tandt/truck
图像数: 251 张，分辨率 979x546
COLMAP: data/tandt/truck/sparse/0/{cameras.bin,images.bin,points3D.bin}
划分: --eval + llffhold=8 → train 219 / test 32
```

## 7. 随机种子

- 3DGS `train.py`：`safe_state()` 内固定 `random.seed(0)` / `np.random.seed(0)` / `torch.manual_seed(0)` / `torch.cuda.set_device("cuda:0")`
- CUDA 光栅化（diff-gaussian-rasterization）的原子累加本身**非确定**，因此逐次运行会有微小数值差异，报告中已注明。

## 8. 环境变量

见 `/scratch/users/nus/e1351071/test_zju/env.sh`（每个 shell 先 `source`）。关键项：`PIP_CONSTRAINT` 锁死 torch/numpy 版本，`TORCH_CUDA_ARCH_LIST=8.0`（A100 sm_80），`PIP_CONFIG_FILE` 屏蔽容器内不可达的 NGC 源，`LD_LIBRARY_PATH` 挂 `tools/x11libs/lib`（容器缺 libX11/libXext，open3d 需要），HF/TORCH 缓存全部隔离到项目内。
