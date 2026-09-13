#!/bin/bash
# 阶段1 后半：torch 之后的依赖安装 + CUDA 扩展编译
set -x
source /scratch/e1351071/zju_test/env.sh
date
echo "=== [0] torch 复验 ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'abi',torch._C._GLIBCXX_USE_CXX11_ABI)" || exit 1

echo "=== [1] numpy<2 + ninja ==="
# 2024 年代 SuGaR/3DGS/open3d 生态按 numpy 1.x 构建，锁 numpy<2 规避 ABI 冲突
printf 'torch==2.4.1\ntorchvision==0.19.1\ntorchaudio==2.4.1\nnumpy<2\n' > $VIRTUAL_ENV/constraints.txt
pip install "numpy<2" ninja --retries 10 --timeout 60 || exit 1
python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)" || exit 1

echo "=== [2] pytorch3d wheel (--no-deps) + 其运行依赖 ==="
pip install --no-deps /scratch/e1351071/zju_test/wheels/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl || exit 1
pip install fvcore iopath --retries 10 --timeout 60 || exit 1

echo "=== [3] SuGaR python 依赖 ==="
pip install open3d plyfile rich tqdm plotly scipy scikit-learn opencv-python --retries 10 --timeout 60 || exit 1
python -c "import torch;print('torch after deps:',torch.__version__, torch.version.cuda, torch.cuda.is_available())" || exit 1

echo "=== [4] 编译 diff-gaussian-rasterization ==="
cd /scratch/e1351071/zju_test/repo/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation --no-deps . || exit 1

echo "=== [5] 编译 simple-knn ==="
cd /scratch/e1351071/zju_test/repo/SuGaR/gaussian_splatting/submodules/simple-knn
TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation --no-deps . || exit 1

echo "=== [6] 最终复验 ==="
cd /scratch/e1351071/zju_test
python -c "import torch;print('FINAL torch',torch.__version__,torch.version.cuda,torch.cuda.is_available())"
date
echo "=== SETUP_STEP2_DONE ==="
