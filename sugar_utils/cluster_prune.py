"""
[ADDED - M2-B] DBSCAN cluster pruning of Gaussian centers, as a *stand-alone*
pre-processing step applied to a trained coarse SuGaR model.

Origin of the idea (see notes/SURVEY2_sugar_direct_followups.md, section 2.3):
    prajwalcr/2d-sugar -> gaussian_splatting_2d/scene/gaussian_model.py:429
        def cluster_gaussians(self, save_path, opt_args):
            def estimate_eps(points, min_samples=6, percentile=90):
                nbrs = NearestNeighbors(n_neighbors=min_samples).fit(points)
                distances, _ = nbrs.kneighbors(points)
                k_distances = distances[:, -1]   # distance to the k-th neighbour
                eps = np.percentile(k_distances, percentile)
                return eps
            points = self._xyz.detach().cpu().numpy()
            eps = estimate_eps(points, min_samples=..., percentile=...)
            cluster_ids = DBSCAN(eps=eps, min_samples=...).fit_predict(points)
    prajwalcr/2d-sugar -> gaussian_splatting_2d/train.py:140-151
            retained_cluster_id = max(cluster_sizes.items(), key=lambda it: it[1])[0]
            gaussians.prune_points(cluster_ids != retained_cluster_id)

Our modifications (M2 + M2+b):
  * rule='keep_large' (default): keep EVERY cluster whose size is >= a fraction of
    the total number of points, instead of only the single largest one.  Only small
    clusters and DBSCAN noise (label -1) are pruned.  The original "only the largest
    cluster survives" behaviour is still available as rule='largest', and is kept
    around as the failure-case control (it deletes the whole background of scenes
    such as Truck, where the background is its own connected component).
  * separate_fg_bg (default True): the foreground (inside the camera bbox, the same
    fg_bbox_factor=1 convention as sugar_extractors/coarse_mesh.py) is much denser
    than the background, so a single global eps estimated on all points is dominated
    by the foreground and is far too small for the background -> the background gets
    labelled as noise and wiped out.  We therefore estimate eps and run DBSCAN
    independently inside and outside the foreground bbox.

NOTE on prune_points() semantics (sugar_scene/sugar_model.py:819): the mask is a
KEEP mask (True = keep); cf. drop_low_opacity_points(), which calls it with
`strengths > threshold`.  All masks returned by this module are KEEP masks.
"""

import time

import numpy as np

try:  # torch is optional for the pure-numpy unit test path
    import torch
except Exception:  # pragma: no cover
    torch = None

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------------------
# eps estimation -- copied from 2D-SuGaR's estimate_eps(), with an added subsampling step
# (the kNN graph of 400k+ points is what makes the original slow / memory hungry).
# --------------------------------------------------------------------------------------
def estimate_eps(points, min_samples=6, percentile=90, subsample=200_000, seed=0):
    """Returns (eps, n_points_used). `points`: (N, 3) float array."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return 0.0, n
    k = int(min(min_samples, n))
    if subsample is not None and 0 < subsample < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=int(subsample), replace=False)
        pts = points[idx]
    else:
        pts = points
    nbrs = NearestNeighbors(n_neighbors=min(k, len(pts))).fit(pts)
    distances, _ = nbrs.kneighbors(pts)
    k_distances = distances[:, -1]  # distance to the k-th neighbour
    eps = float(np.percentile(k_distances, percentile))
    return eps, int(len(pts))


def _cluster_and_keep(points, rule, min_samples, knn_percentile, keep_cluster_min_frac,
                      eps_estimate_subsample, seed=0, n_jobs=-1, eps=None):
    """DBSCAN on `points` (N,3) -> (keep_mask (N,) bool, info dict).  KEEP mask."""
    n = len(points)
    info = {'n_points': int(n)}
    if n == 0:
        return np.zeros(0, dtype=bool), info
    if eps is None:
        eps, n_eps = estimate_eps(points, min_samples=min_samples,
                                  percentile=knn_percentile,
                                  subsample=eps_estimate_subsample, seed=seed)
        info['n_points_for_eps_estimate'] = n_eps
    info['eps'] = float(eps)
    if eps <= 0:
        # degenerate (duplicated points): keep everything
        info['n_clusters'] = 1
        info['n_clusters_kept'] = 1
        info['n_noise'] = 0
        info['cluster_sizes_top10'] = [int(n)]
        return np.ones(n, dtype=bool), info

    t0 = time.time()
    labels = DBSCAN(eps=eps, min_samples=int(min_samples), n_jobs=n_jobs).fit_predict(points)
    info['dbscan_seconds'] = float(time.time() - t0)

    uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
    info['n_clusters'] = int(len(uniq))
    info['n_noise'] = int((labels < 0).sum())
    order = np.argsort(-counts)
    info['cluster_sizes_top10'] = [int(c) for c in counts[order][:10]]

    if len(uniq) == 0:
        # everything is noise -> refuse to delete the whole thing
        info['n_clusters_kept'] = 0
        info['warning'] = 'DBSCAN labelled every point as noise; keeping all points.'
        return np.ones(n, dtype=bool), info

    if rule == 'largest':
        kept_ids = {int(uniq[order[0]])}
    elif rule == 'keep_large':
        thr = float(keep_cluster_min_frac) * float(n)
        kept_ids = set(int(c) for c, s in zip(uniq, counts) if s >= thr)
        if not kept_ids:  # threshold above even the largest cluster
            kept_ids = {int(uniq[order[0]])}
        info['min_cluster_size_threshold'] = float(thr)
    else:
        raise ValueError(f"Unknown rule '{rule}' (expected 'largest' or 'keep_large').")

    info['n_clusters_kept'] = int(len(kept_ids))
    keep = np.isin(labels, list(kept_ids))
    return keep, info


def cluster_prune_mask(
    points,
    fg_bbox_min=None,
    fg_bbox_max=None,
    opacities=None,
    rule='keep_large',
    separate_fg_bg=True,
    min_samples=6,
    knn_percentile=90,
    keep_cluster_min_frac=0.005,
    eps_estimate_subsample=200_000,
    opacity_min=0.0,
    seed=0,
    verbose=True,
):
    """Compute the DBSCAN keep-mask for a set of Gaussian centers.

    Args:
        points: (N, 3) numpy array (or torch tensor) of Gaussian centers.
        fg_bbox_min / fg_bbox_max: (3,) arrays. Required if separate_fg_bg=True.
        opacities: (N,) array of opacities (SuGaR `strengths[..., 0]`), or None.
        rule: 'keep_large' (all clusters >= keep_cluster_min_frac * N) or
              'largest' (2D-SuGaR original: only the biggest cluster).
        opacity_min: points with opacity < this value do NOT take part in the
              clustering but are never pruned either (they are kept as-is).

    Returns:
        keep_mask: (N,) bool numpy array, True = KEEP (prune_points() convention).
        stats: dict, JSON-serialisable.
    """
    t_all = time.time()
    if torch is not None and isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    n_total = len(points)

    stats = {
        'n_total': int(n_total),
        'rule': rule,
        'separate_fg_bg': bool(separate_fg_bg),
        'min_samples': int(min_samples),
        'knn_percentile': float(knn_percentile),
        'keep_cluster_min_frac': float(keep_cluster_min_frac),
        'eps_estimate_subsample': int(eps_estimate_subsample) if eps_estimate_subsample else None,
        'opacity_min': float(opacity_min),
    }

    # Points that participate in the clustering
    if opacities is not None and opacity_min > 0:
        if torch is not None and isinstance(opacities, torch.Tensor):
            opacities = opacities.detach().cpu().numpy()
        active = np.asarray(opacities).reshape(-1) >= opacity_min
    else:
        active = np.ones(n_total, dtype=bool)
    stats['n_inactive_low_opacity_kept'] = int((~active).sum())

    keep_mask = np.ones(n_total, dtype=bool)

    if separate_fg_bg:
        if fg_bbox_min is None or fg_bbox_max is None:
            raise ValueError("separate_fg_bg=True requires fg_bbox_min and fg_bbox_max.")
        fg_bbox_min = np.asarray(fg_bbox_min, dtype=np.float64).reshape(3)
        fg_bbox_max = np.asarray(fg_bbox_max, dtype=np.float64).reshape(3)
        in_fg = np.all(points > fg_bbox_min, axis=-1) & np.all(points < fg_bbox_max, axis=-1)
        stats['fg_bbox_min'] = fg_bbox_min.tolist()
        stats['fg_bbox_max'] = fg_bbox_max.tolist()
        stats['n_fg'] = int(in_fg.sum())
        stats['n_bg'] = int((~in_fg).sum())
        groups = [('fg', in_fg & active), ('bg', (~in_fg) & active)]
    else:
        groups = [('all', active)]

    for name, sel in groups:
        idx = np.flatnonzero(sel)
        if len(idx) == 0:
            stats[name] = {'n_points': 0}
            continue
        keep_sub, info = _cluster_and_keep(
            points[idx], rule=rule, min_samples=min_samples,
            knn_percentile=knn_percentile,
            keep_cluster_min_frac=keep_cluster_min_frac,
            eps_estimate_subsample=eps_estimate_subsample, seed=seed)
        keep_mask[idx] = keep_sub
        info['n_pruned'] = int((~keep_sub).sum())
        stats[name] = info
        if verbose:
            print(f"[cluster_prune] group={name}: n={info['n_points']} eps={info.get('eps'):.6g} "
                  f"clusters={info.get('n_clusters')} kept={info.get('n_clusters_kept')} "
                  f"noise={info.get('n_noise')} pruned={info['n_pruned']} "
                  f"({info['n_pruned'] / max(info['n_points'], 1) * 100:.2f}%)")

    n_pruned = int((~keep_mask).sum())
    stats['n_pruned'] = n_pruned
    stats['n_kept'] = int(n_total - n_pruned)
    stats['pruned_ratio'] = float(n_pruned) / max(n_total, 1)
    stats['total_seconds'] = float(time.time() - t_all)
    stats['warning_high_prune_ratio'] = bool(stats['pruned_ratio'] > 0.30)
    if stats['warning_high_prune_ratio'] and verbose:
        print(f"[cluster_prune] WARNING: pruned {stats['pruned_ratio'] * 100:.2f}% of the "
              f"Gaussians (>30%). eps is probably too small and is cutting the main body "
              f"apart -- increase --knn_percentile or --min_samples, or check "
              f"--separate_fg_bg.")
    if verbose:
        print(f"[cluster_prune] total: {n_total} -> {stats['n_kept']} "
              f"(pruned {n_pruned}, {stats['pruned_ratio'] * 100:.2f}%) "
              f"in {stats['total_seconds']:.1f}s")
    return keep_mask, stats
