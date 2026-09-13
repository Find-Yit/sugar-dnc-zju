#!/usr/bin/env python3
"""
Self-check for the depth -> normal machinery used by the DNC regularizer
(plan section 2, acceptance item C2).

Everything here runs on CPU in a couple of seconds and needs only torch + pytorch3d +
``sugar_utils/dnc_utils.py``.  No SuGaR model, no scene, no CUDA rasterizer.

Run:
    python scripts/test_dnc_normal.py                     # uses repo/SuGaR_dev
    python scripts/test_dnc_normal.py --repo <path>       # any SuGaR checkout
    python scripts/test_dnc_normal.py --device cuda:1

Exit code 0 == all tests passed.  A JSON summary is written next to the script
(``outputs/metrics/test_dnc_normal.json`` by default) so the numbers are on disk.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device):
    """Rebuild the exact PyTorch3D camera SuGaR creates (see sugar_scene/cameras.py:250-325).

    cx = W/2, cy = H/2, scale = min(W, H)/2, principal point offset therefore 0.
    R = I, T = 0 (world == view), which isolates the intrinsics under test.
    """
    from pytorch3d.renderer import FoVPerspectiveCameras
    from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix

    scale = min(W, H) / 2.0
    focal_p3d = torch.tensor([[fx_pix / scale, fy_pix / scale]], dtype=torch.float32)
    p0_p3d = torch.zeros(1, 2, dtype=torch.float32)
    K = _get_sfm_calibration_matrix(1, "cpu", focal_p3d, p0_p3d, orthographic=False)
    R = torch.eye(3, dtype=torch.float32)[None]
    T = torch.zeros(1, 3, dtype=torch.float32)
    return FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device),
                                 K=K.to(device), znear=0.0001)


def synthetic_plane_depth(H, W, normal, point_on_plane, fx, fy, px, py, device, dnc):
    """Depth map (view-space z) of a plane, in the PyTorch3D NDC convention.

    ``normal`` must have a negative z component (plane facing the camera);
    ``point_on_plane`` must lie in front of the camera.
    """
    n = torch.as_tensor(normal, dtype=torch.float32, device=device)
    n = n / n.norm()
    p0 = torch.as_tensor(point_on_plane, dtype=torch.float32, device=device)
    c = (n * p0).sum()

    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)
    # unit-z ray through each pixel
    rx = (x_ndc - px) / fx
    ry = (y_ndc - py) / fy
    r = torch.stack([rx, ry, torch.ones_like(rx)], dim=-1)          # (H, W, 3)
    denom = (r * n).sum(dim=-1)
    t = c / denom
    depth = t * r[..., 2]                                           # == t (r_z == 1)
    return depth, n


def cosine(a, b, dim=-1):
    a = a / a.norm(dim=dim, keepdim=True).clamp(min=1e-12)
    b = b / b.norm(dim=dim, keepdim=True).clamp(min=1e-12)
    return (a * b).sum(dim=dim)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
RESULTS = []


def record(name, passed, detail):
    RESULTS.append({"test": name, "pass": bool(passed), **detail})
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {detail}")
    return passed


def test_pixel_grid_roundtrip(dnc, device, H=546, W=980, fx_pix=1100.0, fy_pix=1100.0):
    """The NDC pixel grid must invert SuGaR's own forward projection exactly."""
    cam = build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device)
    fx, fy, px, py = dnc.get_ndc_intrinsics(cam)
    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)

    vs = torch.tensor([0, 1, 2, H // 3, H // 2, H - 3, H - 2, H - 1], device=device)
    us = torch.tensor([0, 1, 2, W // 3, W // 2, W - 3, W - 2, W - 1], device=device)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    depth = 2.0 + 0.01 * uu.float() + 0.02 * vv.float()

    xs = x_ndc[vv, uu]
    ys = y_ndc[vv, uu]
    pts = torch.stack([(xs - px) * depth / fx, (ys - py) * depth / fy, depth], dim=-1)

    # SuGaR forward projection, verbatim from SuGaR.get_points_depth_in_depth_map
    proj = cam.get_projection_transform().transform_points(pts.reshape(1, -1, 3))[0]
    factor = -1.0 * min(H, W)
    gx = factor / W * proj[..., 0]
    gy = factor / H * proj[..., 1]
    u_rec = (gx + 1.0) / 2.0 * W - 0.5
    v_rec = (gy + 1.0) / 2.0 * H - 0.5

    err_u = (u_rec - uu.reshape(-1).float()).abs().max().item()
    err_v = (v_rec - vv.reshape(-1).float()).abs().max().item()
    return record("pixel_grid_roundtrip_vs_get_points_depth_in_depth_map",
                  err_u < 1e-2 and err_v < 1e-2,
                  {"max_abs_pixel_err_u": err_u, "max_abs_pixel_err_v": err_v, "tol": 1e-2})


def test_unproject_matches_pytorch3d(dnc, device, H=120, W=200, fx_pix=250.0, fy_pix=250.0):
    """Our analytic unprojection must equal FoVPerspectiveCameras.unproject_points."""
    cam = build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device)
    fx, fy, px, py = dnc.get_ndc_intrinsics(cam)
    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)
    torch.manual_seed(0)
    depth = 1.5 + torch.rand(H, W, device=device)

    ours = dnc.unproject_depth_ndc(depth, fx, fy, px, py, x_ndc=x_ndc, y_ndc=y_ndc)
    xy_depth = torch.stack([x_ndc, y_ndc, depth], dim=-1).reshape(1, -1, 3)
    ref = cam.unproject_points(xy_depth, world_coordinates=False)[0].reshape(H, W, 3)

    err = (ours - ref).abs().max().item()
    scale = ref.abs().max().item()
    return record("unproject_matches_pytorch3d_unproject_points",
                  err < 1e-4 * max(scale, 1.0),
                  {"max_abs_err": err, "value_scale": scale, "tol": 1e-4 * max(scale, 1.0)})


def test_plane_normals_ndc(dnc, device, H=200, W=300, fx_pix=400.0, fy_pix=400.0):
    """Known tilted planes -> depth map -> normals; cos with ground truth must be > 0.99."""
    cam = build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device)
    fx, fy, px, py = dnc.get_ndc_intrinsics(cam)
    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)

    cases = [
        (0.0, 0.0, -1.0),      # fronto-parallel
        (0.3, 0.0, -1.0),
        (0.0, 0.4, -1.0),
        (0.5, -0.6, -1.0),
        (-0.7, 0.2, -1.0),
        (1.2, 0.9, -1.0),      # strongly slanted
    ]
    worst_cos, worst_absmin, detail = 1.0, 1.0, []
    for nx, ny, nz in cases:
        depth, n_gt = synthetic_plane_depth(H, W, (nx, ny, nz), (0.0, 0.0, 4.0),
                                            fx, fy, px, py, device, dnc)
        n_d = dnc.depth_to_view_normals(depth, fx, fy, px, py,
                                        x_ndc=x_ndc, y_ndc=y_ndc, orient="towards_camera")
        inner = n_d[2:-2, 2:-2, :]
        cos = cosine(inner, n_gt.view(1, 1, 3))
        cmin = cos.min().item()
        absmin = cos.abs().min().item()
        worst_cos = min(worst_cos, cmin)
        worst_absmin = min(worst_absmin, absmin)
        detail.append({"n_gt": [round(float(v), 4) for v in n_gt.tolist()],
                       "min_cos": cmin, "mean_cos": cos.mean().item()})
    return record("plane_normals_ndc_convention",
                  worst_cos > 0.99,
                  {"worst_min_cos": worst_cos, "worst_min_abs_cos": worst_absmin,
                   "threshold": 0.99, "cases": detail})


def test_plane_normals_pinhole(dnc, device, H=200, W=300):
    """Same check through the OpenCV-style pinhole helper (+X right, +Y down)."""
    fx_pix, fy_pix, cx, cy = 400.0, 400.0, W / 2.0, H / 2.0
    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    rx = (uu + 0.5 - cx) / fx_pix
    ry = (vv + 0.5 - cy) / fy_pix
    r = torch.stack([rx, ry, torch.ones_like(rx)], dim=-1)

    worst, detail = 1.0, []
    for nvec in [(0.0, 0.0, -1.0), (0.4, 0.2, -1.0), (-0.6, 0.5, -1.0), (1.0, -0.8, -1.0)]:
        n = torch.tensor(nvec, dtype=torch.float32, device=device)
        n = n / n.norm()
        c = (n * torch.tensor([0.0, 0.0, 5.0], device=device)).sum()
        depth = c / (r * n).sum(dim=-1)
        pts = dnc.unproject_depth_pinhole(depth, fx_pix, fy_pix, cx, cy)
        n_d = dnc.points_to_normals(pts, orient="towards_camera")
        cos = cosine(n_d[2:-2, 2:-2, :], n.view(1, 1, 3))
        worst = min(worst, cos.min().item())
        detail.append({"n_gt": [round(float(x), 4) for x in n.tolist()],
                       "min_cos": cos.min().item()})
    return record("plane_normals_pinhole_convention", worst > 0.99,
                  {"worst_min_cos": worst, "threshold": 0.99, "cases": detail})


def test_loss_values(dnc, device, H=160, W=240, fx_pix=320.0, fy_pix=320.0):
    """L_dnc must be ~0 for perfectly consistent normals and ~1 for orthogonal ones."""
    cam = build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device)
    fx, fy, px, py = dnc.get_ndc_intrinsics(cam)
    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)
    depth, n_gt = synthetic_plane_depth(H, W, (0.4, -0.3, -1.0), (0.0, 0.0, 4.0),
                                        fx, fy, px, py, device, dnc)
    max_depth = depth.max() * 10.0  # nothing is background here

    n_perfect = n_gt.view(1, 1, 3).expand(H, W, 3).contiguous()
    l_perfect, aux = dnc.depth_normal_consistency_loss(
        depth, n_perfect, fx, fy, px, py, max_depth=max_depth, return_aux=True)

    n_flipped = -n_perfect                       # |cos| must ignore the sign flip
    l_flipped = dnc.depth_normal_consistency_loss(
        depth, n_flipped, fx, fy, px, py, max_depth=max_depth)

    # an in-plane (orthogonal) direction
    tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
    t1 = torch.cross(n_gt, tmp, dim=-1)
    t1 = t1 / t1.norm()
    n_orth = t1.view(1, 1, 3).expand(H, W, 3).contiguous()
    l_orth = dnc.depth_normal_consistency_loss(
        depth, n_orth, fx, fy, px, py, max_depth=max_depth)

    ok = (l_perfect.item() < 1e-3) and (abs(l_flipped.item() - l_perfect.item()) < 1e-6) \
        and (abs(l_orth.item() - 1.0) < 1e-3)
    return record("loss_values_perfect_flipped_orthogonal", ok,
                  {"L_perfect": l_perfect.item(), "L_flipped(-N)": l_flipped.item(),
                   "L_orthogonal": l_orth.item(),
                   "valid_ratio": float(aux["valid_ratio"]),
                   "n_valid": int(aux["n_valid"].item())})


def test_mask_and_gradients(dnc, device, H=160, W=240, fx_pix=320.0, fy_pix=320.0):
    """Background fill and depth discontinuities must be masked out; loss must backprop."""
    cam = build_sugar_style_p3d_camera(H, W, fx_pix, fy_pix, device)
    fx, fy, px, py = dnc.get_ndc_intrinsics(cam)
    x_ndc, y_ndc = dnc.make_ndc_pixel_grid(H, W, device)
    depth, n_gt = synthetic_plane_depth(H, W, (0.3, 0.2, -1.0), (0.0, 0.0, 4.0),
                                        fx, fy, px, py, device, dnc)
    max_depth = torch.tensor(50.0, device=device)
    depth = depth.clone()
    depth[:, W // 2:] = max_depth            # right half = background fill
    depth = depth.detach().requires_grad_(True)

    normals = (n_gt.view(1, 1, 3).expand(H, W, 3) * 0.8).contiguous().requires_grad_(True)
    loss, aux = dnc.depth_normal_consistency_loss(
        depth, normals, fx, fy, px, py, max_depth=max_depth, return_aux=True)
    loss.backward()

    m = aux["mask"]
    bg_leak = int(m[:, W // 2:].sum().item())              # background fill -> must be 0
    # Only the single column adjacent to the discontinuity has a central difference that
    # reaches into the background; columns further left are legitimately valid.
    seam_leak = int(m[:, W // 2 - 1].sum().item())         # must be 0
    fg_kept = int(m[:, :W // 2 - 1].sum().item())          # must stay > 0
    finite = bool(torch.isfinite(depth.grad).all() and torch.isfinite(normals.grad).all())
    grad_nonzero = bool(normals.grad.abs().sum().item() > 0)

    ok = (bg_leak == 0) and (seam_leak == 0) and (fg_kept > 0) and finite and grad_nonzero
    return record("mask_background_discontinuity_and_backward", ok,
                  {"mask_pixels_in_background": bg_leak,
                   "mask_pixels_on_seam_column": seam_leak,
                   "mask_pixels_in_foreground": fg_kept,
                   "grads_finite": finite, "normal_grad_nonzero": grad_nonzero,
                   "loss": loss.item()})


def test_zero_valid_pixels(dnc, device, H=40, W=60):
    """Degenerate case: every pixel is background -> loss 0, no NaN."""
    fx = torch.tensor(1.0, device=device)
    fy = torch.tensor(1.0, device=device)
    depth = torch.full((H, W), 10.0, device=device)
    normals = torch.zeros(H, W, 3, device=device)
    loss = dnc.depth_normal_consistency_loss(depth, normals, fx, fy, 0.0, 0.0,
                                             max_depth=torch.tensor(10.0, device=device))
    ok = torch.isfinite(loss).item() and abs(loss.item()) < 1e-12
    return record("zero_valid_pixels_is_safe", ok, {"loss": loss.item()})


def test_transform_normals_equals_rotation(dnc, device):
    """w2v.transform_normals(n) must equal the rotation part of w2v applied to n.

    The training loop rotates the per-Gaussian normals into view space with
    Transform3d.transform_normals (as SuGaR.render_depth_and_normal already does).
    For a rigid world-to-view transform that must coincide with
    ``transform_points(p + n) - transform_points(p)``.  This test uses a non-trivial
    R and T so that a wrong convention cannot hide.
    """
    from pytorch3d.renderer import FoVPerspectiveCameras
    from pytorch3d.transforms import random_rotations

    torch.manual_seed(7)
    R = random_rotations(1).to(device)
    T = torch.tensor([[0.3, -1.2, 4.5]], device=device)
    cam = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.0001)

    n = torch.nn.functional.normalize(torch.randn(512, 3, device=device), dim=-1)
    p = torch.randn(512, 3, device=device)

    w2v = cam.get_world_to_view_transform()
    n_tn = w2v.transform_normals(n)
    n_diff = w2v.transform_points(p + n) - w2v.transform_points(p)

    err = (n_tn - n_diff).abs().max().item()
    norm_err = (n_tn.norm(dim=-1) - 1.0).abs().max().item()
    return record("transform_normals_equals_rigid_rotation",
                  err < 1e-4 and norm_err < 1e-4,
                  {"max_abs_err_vs_point_difference": err,
                   "max_abs_norm_deviation": norm_err, "tol": 1e-4})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/scratch/e1351071/zju_test/repo/SuGaR_dev",
                    help="SuGaR checkout that contains sugar_utils/dnc_utils.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="/scratch/e1351071/zju_test/outputs/metrics/test_dnc_normal.json")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from sugar_utils import dnc_utils as dnc

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"repo   = {args.repo}")
    print(f"module = {dnc.__file__}")
    print(f"device = {device}\n")

    oks = [
        test_pixel_grid_roundtrip(dnc, device),
        test_unproject_matches_pytorch3d(dnc, device),
        test_plane_normals_ndc(dnc, device),
        test_plane_normals_pinhole(dnc, device),
        test_loss_values(dnc, device),
        test_mask_and_gradients(dnc, device),
        test_zero_valid_pixels(dnc, device),
        test_transform_normals_equals_rotation(dnc, device),
    ]
    n_pass = sum(1 for o in oks if o)
    summary = {"n_tests": len(oks), "n_pass": n_pass, "all_pass": n_pass == len(oks),
               "device": str(device), "module": dnc.__file__, "results": RESULTS}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{n_pass}/{len(oks)} tests passed -> {args.out}")
    return 0 if n_pass == len(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
