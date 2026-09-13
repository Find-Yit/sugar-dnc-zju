"""公共工具：路径设置 / COLMAP 读取（含 track 长度）/ 相机空间尺度 / 伪彩色映射。

本文件只新增，不修改 repo/SuGaR 与 repo/SuGaR_dev 下任何文件（仅 import）。
对应计划 §2b 的指标定义。
"""
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- 路径

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")
SUGAR_DIR = os.environ.get("SUGAR_DIR", os.path.join(PROJ_ROOT, "repo", "SuGaR"))
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_sugar_path(sugar_dir: str = None) -> str:
    """把 SuGaR 仓库与其内嵌的 gaussian_splatting 放进 sys.path（绝对路径，不 chdir）。

    注意：SuGaR 的 sugar_scene/gs_model.py 里写了 sys.path.append('./gaussian_splatting')，
    是相对路径；我们不 chdir 到仓库目录（避免产生垃圾文件），所以这里显式加绝对路径。
    """
    sugar_dir = os.path.abspath(sugar_dir or SUGAR_DIR)
    for p in (sugar_dir, os.path.join(sugar_dir, "gaussian_splatting")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return sugar_dir


def ensure_trailing_sep(path: str) -> str:
    """SuGaR 的 load_gs_cameras 用字符串拼接 `gs_output_path + 'cameras.json'`，必须带结尾 '/'。"""
    return path if path.endswith(os.sep) else path + os.sep


# ---------------------------------------------------------------- COLMAP

def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_points3D_binary_with_tracks(path):
    """读 COLMAP sparse/0/points3D.bin，返回 xyz / rgb / error / track_length。

    3DGS 自带的 gaussian_splatting/scene/colmap_loader.read_points3D_binary 会解析但丢弃
    track 长度，而计划 §2b G1 需要按 track 长度 >= 3 过滤，因此在这里重新实现一遍读取
    （二进制布局与 colmap_loader 完全一致：QdddBBBd + Q + ii*track_length）。
    """
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3), dtype=np.float64)
        rgbs = np.empty((num_points, 3), dtype=np.float64)
        errors = np.empty((num_points, 1), dtype=np.float64)
        track_lengths = np.empty((num_points,), dtype=np.int64)
        for p_id in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            xyzs[p_id] = np.array(props[1:4])
            rgbs[p_id] = np.array(props[4:7])
            errors[p_id] = np.array(props[7])
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            _ = _read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            track_lengths[p_id] = track_length
    return xyzs, rgbs, errors, track_lengths


# ---------------------------------------------------------------- 相机尺度

def cameras_spatial_extent(p3d_cameras):
    """与 SuGaR.get_cameras_spatial_extent(return_average_xyz=True) 逐行等价。

    radius = 1.1 * max_i ||c_i - mean(c)||，其中 c_i 为相机中心（训练集相机）。
    """
    camera_centers = p3d_cameras.get_camera_center()
    avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
    half_diagonal = (camera_centers - avg_camera_center).norm(dim=-1).max().item()
    return 1.1 * half_diagonal, avg_camera_center


# ---------------------------------------------------------------- 伪彩色

_TURBO_LUT = None


def turbo_lut():
    """matplotlib 'turbo' 的 256x3 uint8 查找表。

    LUT 由 tools/doc_env 的 matplotlib 3.10.9 离线导出为 turbo_lut.npy，
    这样 ML venv 不需要安装 matplotlib（见 CLAUDE.md §4.1 两套 venv 隔离规则）。
    """
    global _TURBO_LUT
    if _TURBO_LUT is None:
        _TURBO_LUT = np.load(os.path.join(_THIS_DIR, "turbo_lut.npy"))
    return _TURBO_LUT


def colorize(gray, vmin=0.0, vmax=1.0):
    """灰度 (H, W) float -> 伪彩色 (H, W, 3) uint8，超出 [vmin, vmax] 截断。"""
    lut = turbo_lut()
    x = np.asarray(gray, dtype=np.float64)
    denom = max(float(vmax) - float(vmin), 1e-12)
    x = np.clip((x - float(vmin)) / denom, 0.0, 1.0)
    idx = np.clip((x * 255.0).round().astype(np.int64), 0, 255)
    return lut[idx]


def save_png(path, array_uint8):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array_uint8).save(path)


def to_uint8(img_float_hwc):
    """[0, 1] float (H, W, C) -> uint8。"""
    a = np.clip(np.asarray(img_float_hwc, dtype=np.float32), 0.0, 1.0)
    return (a * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- 固定可视化视角

# 计划 §2b 要求固定 4 个测试视角。测试集为 32 张（llffhold=8，按图像文件名排序后 i%8==0）。
# 这里写死测试集内部的顺序号 0 / 8 / 16 / 24（均匀覆盖整条相机轨迹），
# 对应全量 251 张图像中的第 0 / 64 / 128 / 192 张（即 000001 / 000065 / 000129 / 000193.jpg）。
FIXED_TEST_VIEW_INDICES = [0, 8, 16, 24]

# 误差热图的固定色标上限（三组 run 共用，保证可比）。|render - gt| 的通道均值，范围 [0, 1]。
ERROR_HEATMAP_VMAX = 0.25
