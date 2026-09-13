"""[ADDED - M2-B] CPU-only synthetic unit test for sugar_utils/cluster_prune.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sugar_utils.cluster_prune import cluster_prune_mask  # noqa: E402


def make_scene_a(seed=0):
    """One big blob (50k) + 20 far-away tiny clusters (30 pts each) + 200 uniform noise pts."""
    rng = np.random.default_rng(seed)
    big = rng.normal(0, 0.3, size=(50_000, 3))
    labels = np.zeros(len(big), dtype=int)  # 0 = main body
    smalls = []
    for i in range(20):
        c = rng.uniform(-8, 8, size=3)
        c = c + 6.0 * np.sign(c)  # push far away from the main blob
        smalls.append(rng.normal(0, 0.05, size=(30, 3)) + c)
    smalls = np.concatenate(smalls, 0)
    noise = rng.uniform(-15, 15, size=(200, 3))
    pts = np.concatenate([big, smalls, noise], 0)
    labels = np.concatenate([np.zeros(len(big), int),
                             np.ones(len(smalls), int),
                             2 * np.ones(len(noise), int)])
    return pts, labels  # 0=main, 1=small clusters, 2=noise


def make_scene_b(seed=1):
    """Dense foreground + sparse background (the Truck-like case for separate_fg_bg)."""
    rng = np.random.default_rng(seed)
    fg = rng.normal(0, 0.5, size=(40_000, 3))              # dense, inside bbox (|x|<2)
    # sparse background shell, well outside the fg bbox
    dirs = rng.normal(size=(4_000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bg = dirs * rng.uniform(6.0, 12.0, size=(4_000, 1))
    pts = np.concatenate([fg, bg], 0)
    labels = np.concatenate([np.zeros(len(fg), int), np.ones(len(bg), int)])
    bbox_min = np.array([-2., -2., -2.])
    bbox_max = np.array([2., 2., 2.])
    return pts, labels, bbox_min, bbox_max


def report(name, keep, labels, names):
    print(f"  -> {name}: kept {keep.sum()}/{len(keep)} "
          f"(pruned {(~keep).sum()}, {(~keep).mean()*100:.2f}%)")
    for v, nm in enumerate(names):
        sel = labels == v
        if sel.sum():
            print(f"       {nm:<22s} n={sel.sum():6d}  kept={keep[sel].sum():6d}  "
                  f"pruned={(~keep[sel]).sum():6d} ({(~keep[sel]).mean()*100:6.2f}%)")


def main():
    print("=" * 90)
    print("TEST 1 -- big blob + 20 tiny far clusters + 200 uniform noise points")
    print("=" * 90)
    pts, labels = make_scene_a()
    names = ['main body (50000)', 'small clusters (600)', 'uniform noise (200)']
    bbox_min, bbox_max = np.array([-3.] * 3), np.array([3.] * 3)

    for rule in ['keep_large', 'largest']:
        keep, st = cluster_prune_mask(pts, bbox_min, bbox_max, rule=rule,
                                      separate_fg_bg=False, verbose=False)
        print(f"\n[rule={rule}, separate_fg_bg=False] eps={st['all']['eps']:.4g} "
              f"clusters={st['all']['n_clusters']} kept_clusters={st['all']['n_clusters_kept']} "
              f"noise={st['all']['n_noise']} sizes_top5={st['all']['cluster_sizes_top10'][:5]}")
        report(rule, keep, labels, names)
        # NOTE: a Gaussian blob has thin tails, so DBSCAN always labels a few percent of
        # the outermost main-body points as noise. We require >=90% of the main body to
        # survive, and 100% of the small clusters / uniform noise to be removed.
        main_kept = keep[labels == 0].mean()
        ok = (main_kept >= 0.90) and (not keep[labels == 1].any()) and (not keep[labels == 2].any())
        print(f"     EXPECTED: main body >=90% kept (got {main_kept*100:.2f}%), small clusters "
              f"+ noise 100% pruned -> {'PASS' if ok else 'FAIL'}")

    print()
    print("=" * 90)
    print("TEST 2 -- dense foreground (40000) + sparse background shell (4000)")
    print("=" * 90)
    pts, labels, bbox_min, bbox_max = make_scene_b()
    names = ['foreground (dense)', 'background (sparse)']
    for sep in [True, False]:
        keep, st = cluster_prune_mask(pts, bbox_min, bbox_max, rule='keep_large',
                                      separate_fg_bg=sep, verbose=False)
        if sep:
            print(f"\n[separate_fg_bg=True]  eps_fg={st['fg']['eps']:.4g} "
                  f"(clusters={st['fg']['n_clusters']}, noise={st['fg']['n_noise']})  "
                  f"eps_bg={st['bg']['eps']:.4g} "
                  f"(clusters={st['bg']['n_clusters']}, noise={st['bg']['n_noise']})")
        else:
            print(f"\n[separate_fg_bg=False] eps_global={st['all']['eps']:.4g} "
                  f"clusters={st['all']['n_clusters']} noise={st['all']['n_noise']} "
                  f"sizes_top5={st['all']['cluster_sizes_top10'][:5]}")
        report(f"separate_fg_bg={sep}", keep, labels, names)
        bg_pruned = (~keep[labels == 1]).mean()
        print(f"     background pruned ratio = {bg_pruned*100:.2f}%  "
              f"| total pruned {st['pruned_ratio']*100:.2f}% "
              f"| >30% warning flag = {st['warning_high_prune_ratio']}")
        if sep:
            print(f"     EXPECTED: background survives -> "
                  f"{'PASS' if bg_pruned < 0.10 else 'FAIL'}")
        else:
            print(f"     EXPECTED: global eps wipes out the sparse background -> "
                  f"{'PASS (background destroyed, as predicted)' if bg_pruned > 0.5 else 'NOTE: background survived even without the fg/bg split'}")


if __name__ == '__main__':
    main()
