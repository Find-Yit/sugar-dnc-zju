#!/usr/bin/env python
"""碎片分量着色图：open3d cluster_connected_triangles + pytorch3d MeshRasterizer。

相机构建与 scripts/eval/render_mesh_views.py 完全一致（load_gs_cameras + CamerasWrapper，
测试集 interval=8，视角 0/8/16/24）。着色规则：
  * 面数 <100 的碎片分量 -> 醒目红 (230,30,30)
  * 最大分量 -> 灰 (170,170,170)
  * 其余分量 -> 按分量 id 随机浅色
背景白。
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.environ["PROJ_ROOT"], "repo/sugar-dnc-zju/scripts/eval"))
from _common import ensure_trailing_sep, save_png, setup_sugar_path

p = argparse.ArgumentParser()
p.add_argument("--scene_path", required=True)
p.add_argument("--gs_checkpoint", required=True)
p.add_argument("--mesh_path", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--frag_thresh", type=int, default=100)
args = p.parse_args()

setup_sugar_path(None)
import open3d as o3d, torch
torch.cuda.set_device(args.gpu)
device = torch.device(f"cuda:{args.gpu}")
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from sugar_scene.cameras import CamerasWrapper, load_gs_cameras

t0 = time.time()
cam_list = load_gs_cameras(source_path=args.scene_path,
                           gs_output_path=ensure_trailing_sep(args.gs_checkpoint),
                           load_gt_images=True)
test_cams = [c for i, c in enumerate(cam_list) if i % 8 == 0]
test_cameras = CamerasWrapper(test_cams)
H, W = int(test_cams[0].image_height), int(test_cams[0].image_width)

m = o3d.io.read_triangle_mesh(args.mesh_path)
verts_np = np.asarray(m.vertices, dtype=np.float32)
faces_np = np.asarray(m.triangles, dtype=np.int64)
lab, n_tri, _ = m.cluster_connected_triangles()
lab = np.asarray(lab); n_tri = np.asarray(n_tri)
n_comp = len(n_tri)
largest = int(np.argmax(n_tri))
n_frag = int((n_tri < args.frag_thresh).sum())
print(f"[{args.tag}] faces={len(faces_np)} comps={n_comp} frags(<{args.frag_thresh})={n_frag} largest={n_tri[largest]}")

rng = np.random.default_rng(0)
comp_col = (rng.integers(150, 235, size=(n_comp, 3))).astype(np.uint8)  # 浅色
comp_col[n_tri < args.frag_thresh] = np.array([230, 30, 30], dtype=np.uint8)
comp_col[largest] = np.array([170, 170, 170], dtype=np.uint8)
face_col = torch.from_numpy(comp_col[lab].astype(np.uint8)).to(device)

verts = torch.from_numpy(verts_np).to(device)
faces = torch.from_numpy(faces_np).to(device)
meshes = Meshes(verts=[verts], faces=[faces])
rs = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
                           max_faces_per_bin=200000)
os.makedirs(args.out_dir, exist_ok=True)
stats = {"tag": args.tag, "n_components": n_comp, "n_fragments_lt100": n_frag,
         "largest_component_faces": int(n_tri[largest]),
         "largest_component_ratio": float(n_tri[largest] / len(faces_np)),
         "frag_pixel_ratio": {}}
with torch.no_grad():
    for idx in [0, 8, 16, 24]:
        cam = test_cameras.p3d_cameras[idx]
        frags = MeshRasterizer(cameras=cam, raster_settings=rs)(meshes)
        ptf = frags.pix_to_face[0, ..., 0]
        hit = ptf >= 0
        safe = torch.where(hit, ptf, torch.zeros_like(ptf))
        rgb = face_col[safe]
        rgb = torch.where(hit[..., None], rgb, torch.full_like(rgb, 255))
        arr = rgb.cpu().numpy()
        isred = ((arr[..., 0] == 230) & (arr[..., 1] == 30)).mean()
        stats["frag_pixel_ratio"][f"view{idx:02d}"] = float(isred)
        save_png(os.path.join(args.out_dir, f"fragments_{args.tag}_view{idx:02d}.png"), arr)
        print(f"  view{idx:02d} 碎片像素占比 {isred*100:.3f}%")
        del frags; torch.cuda.empty_cache()
json.dump(stats, open(os.path.join(args.out_dir, f"fragstats_{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
print("耗时", round(time.time() - t0, 1), "s")
