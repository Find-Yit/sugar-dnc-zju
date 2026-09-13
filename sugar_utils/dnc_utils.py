"""
dnc_utils.py -- Depth-Normal Consistency (DNC) regularization utilities for SuGaR.

This module is intentionally *self-contained*: it only depends on ``torch``,
``numpy`` and ``PIL`` so that every function can be imported and unit-tested
without instantiating a SuGaR model, a scene or a CUDA rasterizer.

--------------------------------------------------------------------------
Coordinate / convention contract (IMPORTANT -- read before editing)
--------------------------------------------------------------------------
SuGaR builds its cameras with ``sugar_scene.cameras.convert_camera_from_gs_to_pytorch3d``,
which returns a ``pytorch3d.renderer.FoVPerspectiveCameras`` whose ``K`` is produced by
``pytorch3d.utils._get_sfm_calibration_matrix`` (column-major):

        K = [[fx, 0,  px, 0],
             [0,  fy, py, 0],
             [0,  0,  0,  1],
             [0,  0,  1,  0]]

with ``fx = fx_pixels / s``, ``fy = fy_pixels / s``, ``s = min(W, H) / 2`` and
``px = py = 0`` (SuGaR uses cx = W/2, cy = H/2, so the principal point offset vanishes).

``FoVPerspectiveCameras.get_projection_transform`` uses this ``K`` verbatim, therefore the
view-space -> NDC map is simply

        x_ndc = fx * x_view / z_view + px
        y_ndc = fy * y_view / z_view + py

so the *unprojection* used here is the exact analytic inverse

        x_view = (x_ndc - px) * z_view / fx
        y_view = (y_ndc - py) * z_view / fy
        z_view = z_view                                   (= the rendered depth)

**View space** is the PyTorch3D camera space: **+X points left, +Y points up,
+Z points forward, away from the camera**.  Depth is therefore *positive* in front
of the camera, and a normal that faces the camera has ``n_z <= 0``.

**Pixel -> NDC.**  The pixel grid below is the exact inverse of SuGaR's own forward
projection ``SuGaR.get_points_depth_in_depth_map`` (sugar_scene/sugar_model.py), which does::

        factor = -1 * min(H, W)
        grid_x = factor / W * x_ndc      # grid for F.grid_sample
        grid_y = factor / H * y_ndc

and ``F.grid_sample`` maps ``grid_x = 2 (u + 0.5) / W - 1`` to column ``u``.  Inverting:

        x_ndc(u) = -(2 (u + 0.5) / W - 1) * W / min(H, W)
        y_ndc(v) = -(2 (v + 0.5) / H - 1) * H / min(H, W)

(u = column index growing to the right, v = row index growing downward).  Note that the
*shorter* image side spans [-1, 1] in NDC and the longer side spans [-a, a] with
a = max(W, H) / min(W, H) -- the standard PyTorch3D NDC convention.  ``test_dnc_normal.py``
verifies this round-trip numerically against ``get_points_depth_in_depth_map``'s formula and
against ``FoVPerspectiveCameras.unproject_points``.

--------------------------------------------------------------------------
The loss
--------------------------------------------------------------------------
Let ``N`` be the alpha-composited per-pixel Gaussian normal (rendered in view space) and
``N_d`` the normal obtained from the rendered depth map by central differences of the
unprojected point map:

        P(u, v)   = unproject(x_ndc(u), y_ndc(v), D(u, v))            in view space
        dP/du     = (P(v, u+1) - P(v, u-1)) / 2
        dP/dv     = (P(v+1, u) - P(v-1, u)) / 2
        N_d       = normalize(dP/du  x  dP/dv)

        L_dnc = mean_{valid pixels} ( 1 - |cos(N, N_d)| )

The absolute value removes the sign ambiguity of the smallest-scale axis of a 3D Gaussian
(a 3D ellipsoid has no intrinsic normal orientation, unlike a 2D Gaussian disk).
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def get_ndc_intrinsics(p3d_camera):
    """Extract (fx, fy, px, py) of the PyTorch3D NDC intrinsics from a FoVPerspectiveCameras.

    Args:
        p3d_camera: a pytorch3d FoVPerspectiveCameras built by SuGaR (must carry a ``K``).

    Returns:
        Tuple of 4 zero-dim torch tensors (fx, fy, px, py) in NDC units.
    """
    K = p3d_camera.K
    if K is None:
        raise ValueError(
            "The provided PyTorch3D camera has no K matrix; DNC needs the NDC intrinsics."
        )
    K = K.reshape(-1, 4, 4)[0]
    return K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def make_ndc_pixel_grid(image_height, image_width, device, dtype=torch.float32):
    """Pixel-center NDC coordinates matching SuGaR's PyTorch3D convention.

    Returns:
        (x_ndc, y_ndc): two tensors of shape (H, W).
    """
    H, W = int(image_height), int(image_width)
    s = float(min(H, W))
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    x_1d = -(2.0 * (u + 0.5) / W - 1.0) * (W / s)
    y_1d = -(2.0 * (v + 0.5) / H - 1.0) * (H / s)
    y_grid, x_grid = torch.meshgrid(y_1d, x_1d, indexing="ij")
    return x_grid, y_grid


# ---------------------------------------------------------------------------
# Depth -> 3D points -> normals  (the core, unit-tested functions)
# ---------------------------------------------------------------------------
def unproject_depth_ndc(depth, fx, fy, px=0.0, py=0.0, x_ndc=None, y_ndc=None):
    """Back-project a depth map into PyTorch3D view-space points.

    Args:
        depth: (H, W) tensor, the *view-space z* of each pixel (positive in front).
        fx, fy, px, py: NDC intrinsics (see :func:`get_ndc_intrinsics`).
        x_ndc, y_ndc: optional precomputed (H, W) NDC pixel grids; built on the fly if None.

    Returns:
        (H, W, 3) tensor of view-space points. Differentiable w.r.t. ``depth``.
    """
    if depth.dim() != 2:
        raise ValueError(f"depth must be (H, W), got {tuple(depth.shape)}")
    H, W = depth.shape
    if x_ndc is None or y_ndc is None:
        x_ndc, y_ndc = make_ndc_pixel_grid(H, W, depth.device, depth.dtype)
    x_view = (x_ndc - px) * depth / fx
    y_view = (y_ndc - py) * depth / fy
    return torch.stack([x_view, y_view, depth], dim=-1)


def unproject_depth_pinhole(depth, fx_pix, fy_pix, cx_pix, cy_pix):
    """Reference OpenCV-style pinhole unprojection (+X right, +Y down, +Z forward).

    Provided for independent cross-checking of :func:`unproject_depth_ndc`; it is *not*
    used by the training loop (SuGaR's cameras live in PyTorch3D NDC space).

    Args:
        depth: (H, W) tensor of z values.
        fx_pix, fy_pix, cx_pix, cy_pix: intrinsics in pixel units.

    Returns:
        (H, W, 3) tensor of camera-space points.
    """
    H, W = depth.shape
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    x = (uu + 0.5 - cx_pix) * depth / fx_pix
    y = (vv + 0.5 - cy_pix) * depth / fy_pix
    return torch.stack([x, y, depth], dim=-1)


def points_to_normals(points, orient=None, eps=1e-12, return_norm=False):
    """Normals of a structured point map via central differences + cross product.

    ``n = normalize( dP/du  x  dP/dv )`` where u is the column (rightwards) and v the row
    (downwards) index.  The 1-pixel border, where the central difference is undefined, is
    filled with zeros.

    Args:
        points: (H, W, 3) structured point map.
        orient: ``None`` -> no re-orientation; ``'towards_camera'`` -> force ``n_z <= 0``
            (PyTorch3D view space, +Z away from the camera);
            ``'away_from_camera'`` -> force ``n_z >= 0``;
            ``'neg_z'`` / ``'pos_z'`` are aliases of the two above.
        eps: norm clamp for the normalization.
        return_norm: if True, also return the (H, W) raw cross-product magnitude.

    Returns:
        (H, W, 3) unit normals (border = 0), and optionally the (H, W) raw magnitude.
    """
    if points.dim() != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be (H, W, 3), got {tuple(points.shape)}")
    dpdu = 0.5 * (points[1:-1, 2:, :] - points[1:-1, :-2, :])
    dpdv = 0.5 * (points[2:, 1:-1, :] - points[:-2, 1:-1, :])
    n = torch.cross(dpdu, dpdv, dim=-1)
    raw = n.norm(dim=-1, keepdim=True)
    n = n / raw.clamp(min=eps)

    if orient in ("towards_camera", "neg_z"):
        n = n * torch.where(n[..., 2:3] > 0, -torch.ones_like(n[..., 2:3]), torch.ones_like(n[..., 2:3]))
    elif orient in ("away_from_camera", "pos_z"):
        n = n * torch.where(n[..., 2:3] < 0, -torch.ones_like(n[..., 2:3]), torch.ones_like(n[..., 2:3]))
    elif orient is not None:
        raise ValueError(f"Unknown orient={orient!r}")

    out = torch.zeros_like(points)
    out[1:-1, 1:-1, :] = n
    if return_norm:
        out_raw = torch.zeros(points.shape[:2], device=points.device, dtype=points.dtype)
        out_raw[1:-1, 1:-1] = raw[..., 0]
        return out, out_raw
    return out


def depth_to_view_normals(depth, fx, fy, px=0.0, py=0.0, x_ndc=None, y_ndc=None,
                          orient="towards_camera"):
    """Convenience composition: depth map -> view-space normals (see the two functions above)."""
    pts = unproject_depth_ndc(depth, fx, fy, px=px, py=py, x_ndc=x_ndc, y_ndc=y_ndc)
    return points_to_normals(pts, orient=orient)


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------
def build_dnc_mask(depth, normal_norm, max_depth, bg_depth_ratio=0.98, border=2,
                   depth_grad_rel_thresh=0.05, min_normal_norm=0.1):
    """Boolean (H, W) mask of pixels on which L_dnc is evaluated.

    Criteria (plan section 2(c)):
      1. ``depth < bg_depth_ratio * max_depth``  -- drops the background fill value,
         eroded by one pixel so that the central differences never touch background;
      2. the ``border`` outermost pixel rings are discarded;
      3. ``max(|dD/du|, |dD/dv|) < depth_grad_rel_thresh * depth`` -- drops pixels that
         straddle a depth discontinuity (object silhouettes), where a finite difference
         would produce a meaningless normal;
      4. (numerical guard, not in the plan) ``||N_raw|| > min_normal_norm`` -- the rendered
         normal map is alpha-composited, so pixels with a low accumulated opacity carry an
         almost-zero, direction-less vector whose normalization is unstable.
    """
    H, W = depth.shape
    device = depth.device
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)

    valid_depth = depth < (bg_depth_ratio * max_depth)
    # (1) eroded foreground: the pixel and its 4 neighbours must all be foreground
    core = (valid_depth[1:-1, 1:-1] & valid_depth[1:-1, 2:] & valid_depth[1:-1, :-2]
            & valid_depth[2:, 1:-1] & valid_depth[:-2, 1:-1])

    # (3) relative depth gradient
    du = 0.5 * (depth[1:-1, 2:] - depth[1:-1, :-2]).abs()
    dv = 0.5 * (depth[2:, 1:-1] - depth[:-2, 1:-1]).abs()
    grad_ok = torch.maximum(du, dv) < (depth_grad_rel_thresh * depth[1:-1, 1:-1].abs())

    mask[1:-1, 1:-1] = core & grad_ok

    # (4) rendered-normal magnitude guard
    if normal_norm is not None:
        mask = mask & (normal_norm > min_normal_norm)

    # (2) border erosion
    b = int(border)
    if b > 0:
        border_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        border_mask[b:H - b, b:W - b] = True
        mask = mask & border_mask
    return mask


# ---------------------------------------------------------------------------
# The loss itself
# ---------------------------------------------------------------------------
def depth_normal_consistency_loss(
    depth,
    normal_view,
    fx, fy, px=0.0, py=0.0,
    max_depth=None,
    bg_depth_ratio=0.98,
    border=2,
    depth_grad_rel_thresh=0.05,
    min_normal_norm=0.1,
    x_ndc=None, y_ndc=None,
    return_aux=False,
):
    """Depth-normal consistency loss ``mean_valid(1 - |cos(N, N_d)|)``.

    Args:
        depth: (H, W) rendered view-space depth (background filled with ``max_depth``).
        normal_view: (H, W, 3) rendered, alpha-composited, *view-space* Gaussian normals
            (not normalized -- its magnitude is used as the opacity guard).
        fx, fy, px, py: NDC intrinsics of the camera the two maps were rendered with.
        max_depth: the background fill value used for ``depth``.
        return_aux: if True, also return a dict with ``n_d``, ``n_pred``, ``mask``,
            ``n_valid`` and ``valid_ratio`` for logging / visualization.

    Returns:
        Zero-dim loss tensor (0.0 with no gradient when no pixel is valid), and optionally
        the aux dict.
    """
    if max_depth is None:
        max_depth = depth.max().detach()

    points = unproject_depth_ndc(depth, fx, fy, px=px, py=py, x_ndc=x_ndc, y_ndc=y_ndc)
    n_d = points_to_normals(points, orient="towards_camera")

    normal_norm = normal_view.norm(dim=-1)
    n_pred = normal_view / normal_norm.unsqueeze(-1).clamp(min=1e-8)

    mask = build_dnc_mask(
        depth, normal_norm.detach(), max_depth,
        bg_depth_ratio=bg_depth_ratio, border=border,
        depth_grad_rel_thresh=depth_grad_rel_thresh, min_normal_norm=min_normal_norm,
    )
    mask_f = mask.to(n_pred.dtype)
    n_valid = mask_f.sum()

    cos = (n_pred * n_d).sum(dim=-1)
    per_pixel = 1.0 - cos.abs()
    loss = (per_pixel * mask_f).sum() / n_valid.clamp(min=1.0)

    if return_aux:
        aux = {
            "n_d": n_d,
            "n_pred": n_pred,
            "mask": mask,
            "n_valid": n_valid,
            "valid_ratio": (n_valid / float(mask.numel())),
        }
        return loss, aux
    return loss


# ---------------------------------------------------------------------------
# Visualization (PIL only -- matplotlib is deliberately NOT a dependency)
# ---------------------------------------------------------------------------
def _to_uint8_gray(arr, lo=None, hi=None):
    arr = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.nanmin(arr))
    if hi is None:
        hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - lo) / (hi - lo)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _normal_to_uint8_rgb(n):
    n = np.asarray(n, dtype=np.float32)
    return (np.clip(n * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_dnc_visualization(out_dir, iteration, depth, normal_pred, normal_from_depth, mask,
                           max_depth=None, save_grid=True):
    """Dump D / N / N_d / mask as PNGs into ``out_dir`` (created if needed).

    Returns the list of written file paths.
    """
    import os
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    def _np(x):
        return x.detach().float().cpu().numpy()

    d = _np(depth)
    m = _np(mask).astype(bool) if mask is not None else None
    npred = _np(normal_pred)
    nd = _np(normal_from_depth)

    # Depth is normalized on the valid (foreground) pixels only, using 1%/99% percentiles
    # so that a few outliers do not flatten the whole visualization.
    if m is not None and m.sum() > 0:
        lo, hi = np.percentile(d[m], [1.0, 99.0])
    else:
        lo, hi = float(d.min()), float(d.max())
    d_img = _to_uint8_gray(d, lo=float(lo), hi=float(hi))

    paths = []
    tag = f"iter{int(iteration):06d}"
    items = [
        (f"{tag}_depth.png", Image.fromarray(d_img, mode="L")),
        (f"{tag}_normal_rendered.png", Image.fromarray(_normal_to_uint8_rgb(npred), mode="RGB")),
        (f"{tag}_normal_from_depth.png", Image.fromarray(_normal_to_uint8_rgb(nd), mode="RGB")),
    ]
    if m is not None:
        items.append((f"{tag}_mask.png", Image.fromarray((m * 255).astype(np.uint8), mode="L")))
    for name, img in items:
        p = os.path.join(out_dir, name)
        img.save(p)
        paths.append(p)

    if save_grid:
        H, W = d.shape
        grid = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)
        grid[:H, :W] = np.stack([d_img] * 3, axis=-1)
        grid[:H, W:] = _normal_to_uint8_rgb(npred)
        grid[H:, :W] = _normal_to_uint8_rgb(nd)
        if m is not None:
            grid[H:, W:] = np.stack([(m * 255).astype(np.uint8)] * 3, axis=-1)
        p = os.path.join(out_dir, f"{tag}_grid.png")
        Image.fromarray(grid, mode="RGB").save(p)
        paths.append(p)
    return paths
