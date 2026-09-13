#!/bin/bash
# 自动生成 notes/ENV.md（阶段 1 验收条目 A5）
source /scratch/e1351071/zju_test/env.sh
OUT=$PROJ_ROOT/notes/ENV.md
{
echo "# ENV.md — 运行环境记录"
echo
echo "生成时间：\`$(date '+%Y-%m-%d %H:%M:%S %Z')\`  |  主机：\`$(hostname)\`  |  PBS 作业：\`${PBS_JOBID:-未知}\`"
echo
echo "## 1. 硬件"
echo
echo '```'
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
echo
echo "CPU 核数: $(nproc)"
echo "内存: $(free -g | awk '/^Mem:/{print $2" GB"}')"
echo "/scratch 可用: $(df -h /scratch | tail -1 | awk '{print $4}')"
echo '```'
echo
echo "## 2. 操作系统与编译器"
echo
echo '```'
grep PRETTY_NAME /etc/os-release
echo "kernel: $(uname -r)"
echo "glibc: $(ldd --version | head -1)"
gcc --version | head -1
/usr/local/cuda/bin/nvcc --version | tail -2
echo '```'
echo
echo "## 3. Python / PyTorch"
echo
echo '```'
echo "python: $(python -c 'import sys;print(sys.version)')"
echo "venv:   $VIRTUAL_ENV"
python - <<'PY'
import torch, torchvision
print("torch:      ", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cudnn:      ", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
print("device 0:   ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GLIBCXX_USE_CXX11_ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY
echo '```'
echo
echo "## 4. 关键依赖版本"
echo
echo '```'
pip list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|pytorch3d|open3d|numpy|scipy|scikit-learn|plyfile|rich|tqdm|plotly|opencv-python|fvcore|iopath|ninja|diff-gaussian-rasterization|simple-knn|pillow|matplotlib|pandas) "
echo '```'
echo
echo "完整冻结清单：\`notes/pip_freeze.txt\`"
echo
echo "## 5. 代码仓库"
echo
echo '```'
cd $PROJ_ROOT/repo/SuGaR
echo "SuGaR repo: $PROJ_ROOT/repo/SuGaR"
echo "commit: $(git log -1 --format='%H')"
echo "date:   $(git log -1 --format='%ci')"
echo "subject: $(git log -1 --format='%s')"
echo "upstream: $(git remote -v | head -1)"
echo "工作区改动: $(git status --porcelain | wc -l) 个文件（阶段1/2 应为 0）"
echo '```'
echo
echo "## 6. 数据"
echo
echo '```'
echo "来源: data/tandt_db.zip (Tanks&Temples + DeepBlending，3DGS 官方发布包)"
echo "大小: $(stat -c%s $PROJ_ROOT/data/tandt_db.zip) 字节"
echo "sha256: $(cat $PROJ_ROOT/notes/tandt_db.sha256 2>/dev/null || echo '(见 notes/tandt_db.sha256)')"
echo "使用场景: data/tandt/truck"
echo "图像数: $(ls $PROJ_ROOT/data/tandt/truck/images | wc -l) 张，分辨率 979x546"
echo "COLMAP: data/tandt/truck/sparse/0/{cameras.bin,images.bin,points3D.bin}"
echo "划分: --eval + llffhold=8 → train 219 / test 32"
echo '```'
echo
echo "## 7. 随机种子"
echo
echo '- 3DGS `train.py`：`safe_state()` 内固定 `random.seed(0)` / `np.random.seed(0)` / `torch.manual_seed(0)` / `torch.cuda.set_device("cuda:0")`'
echo '- CUDA 光栅化（diff-gaussian-rasterization）的原子累加本身**非确定**，因此逐次运行会有微小数值差异，报告中已注明。'
echo
echo "## 8. 环境变量"
echo
echo '见 `'"$PROJ_ROOT"'/env.sh`（每个 shell 先 `source`）。关键项：`PIP_CONSTRAINT` 锁死 torch/numpy 版本，`TORCH_CUDA_ARCH_LIST=9.0`（H200 sm_90），HF/TORCH 缓存全部隔离到项目内。'
} > $OUT
pip freeze > $PROJ_ROOT/notes/pip_freeze.txt
echo "ENV.md written to $OUT"
