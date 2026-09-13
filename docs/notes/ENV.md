# ENV.md — 运行环境记录

生成时间：`2026-09-13 05:49:14 Asia`  |  主机：`hopper-29`  |  PBS 作业：`614050.hopper-m-02`

## 1. 硬件

```
index, name, memory.total [MiB], driver_version, compute_cap
0, NVIDIA H200, 143771 MiB, 575.57.08, 9.0
1, NVIDIA H200, 143771 MiB, 575.57.08, 9.0

CPU 核数: 24
内存: 2015 GB
/scratch 可用: 3.2P
```

## 2. 操作系统与编译器

```
PRETTY_NAME="Ubuntu 22.04.3 LTS"
kernel: 5.14.0-427.13.1.el9_4.x86_64
glibc: ldd (Ubuntu GLIBC 2.35-0ubuntu3.4) 2.35
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

## 3. Python / PyTorch

```
python: 3.10.12 (main, Jan  8 2026, 06:52:19) [GCC 11.4.0]
venv:   /scratch/e1351071/virtualenvs/zju_test
torch:       2.4.1+cu121
torchvision: 0.19.1+cu121
torch.version.cuda: 12.1
cudnn:       90100
cuda available: True | device count: 2
device 0:    NVIDIA H200
GLIBCXX_USE_CXX11_ABI: False
```

## 4. 关键依赖版本

```
diff-gaussian-rasterization 0.0.0
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
simple-knn                  0.0.0
torch                       2.4.1
torchaudio                  2.4.1
torchvision                 0.19.1
tqdm                        4.70.1
```

完整冻结清单：`notes/pip_freeze.txt`

## 5. 代码仓库

```
SuGaR repo: /scratch/e1351071/zju_test/repo/SuGaR
commit: 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44
date:   2024-09-24 12:14:40 -0400
subject: Minor fix.
upstream: origin	https://github.com/Anttwo/SuGaR.git (fetch)
工作区改动: 0 个文件（阶段1/2 应为 0）
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

见 `/scratch/e1351071/zju_test/env.sh`（每个 shell 先 `source`）。关键项：`PIP_CONSTRAINT` 锁死 torch/numpy 版本，`TORCH_CUDA_ARCH_LIST=9.0`（H200 sm_90），HF/TORCH 缓存全部隔离到项目内。
