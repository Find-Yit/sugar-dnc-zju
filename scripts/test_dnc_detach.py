#!/usr/bin/env python
"""Unit check for the plan-§3d switch `--dnc_detach_depth`.

Verifies, on CPU and without any GPU, the exact property the switch is supposed to have:

  detach=False -> dL_dnc/d(depth) != 0   (the loss can "flatten the depth"  = trivial solution)
  detach=True  -> dL_dnc/d(depth) == 0   (gradient reaches the model ONLY through the normal map)

and that in both cases the gradient w.r.t. the normal map stays non-zero (the term still
does its intended job of aligning N with N_d).

Run:  python scripts/test_dnc_detach.py --repo repo/SuGaR_dev
"""
import argparse, json, os, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/scratch/e1351071/zju_test/repo/SuGaR_dev')
    ap.add_argument('--out', default='/scratch/e1351071/zju_test/outputs/metrics/test_dnc_detach.json')
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from sugar_utils import dnc_utils

    torch.manual_seed(0)
    H, W = 48, 64
    fx = fy = 1.5
    # A curved, non-planar depth surface so that N_d genuinely depends on the depth values.
    v, u = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
    z_leaf = (3.0 + 0.4 * u ** 2 + 0.3 * v ** 2 + 0.1 * u * v).clone().requires_grad_(True)
    # A normal map deliberately *misaligned* with the depth geometry, so L_dnc > 0 and both
    # partial derivatives are generically non-zero.
    n_leaf = torch.randn(H, W, 3) * 0.1 + torch.tensor([0.3, 0.2, -0.9])
    n_leaf = n_leaf.clone().requires_grad_(True)
    grid = dnc_utils.make_ndc_pixel_grid(H, W, torch.device('cpu'))
    max_depth = torch.tensor(100.0)          # nothing is masked out as background

    res = {}
    for detach in (False, True):
        if z_leaf.grad is not None:
            z_leaf.grad = None
        if n_leaf.grad is not None:
            n_leaf.grad = None
        depth = z_leaf * 1.0                 # non-leaf, exactly like the rendered depth map
        normal = n_leaf * 1.0
        depth_for_loss = depth.detach() if detach else depth
        loss, aux = dnc_utils.depth_normal_consistency_loss(
            depth_for_loss, normal, fx, fy, 0.0, 0.0,
            max_depth=max_depth, border=2, depth_grad_rel_thresh=1e9,
            min_normal_norm=0.0, x_ndc=grid[0], y_ndc=grid[1], return_aux=True)
        loss.backward()
        gz = 0.0 if z_leaf.grad is None else float(z_leaf.grad.abs().max())
        gn = 0.0 if n_leaf.grad is None else float(n_leaf.grad.abs().max())
        res[f'detach={detach}'] = {
            'loss': float(loss), 'valid_ratio': float(aux['valid_ratio']),
            'max_abs_grad_wrt_depth': gz, 'max_abs_grad_wrt_normal': gn,
            'depth_for_loss_is_depth': depth_for_loss is depth,
            'depth_for_loss_requires_grad': bool(depth_for_loss.requires_grad),
        }

    off, on = res['detach=False'], res['detach=True']
    checks = {
        'loss_identical_both_modes': abs(off['loss'] - on['loss']) < 1e-9,
        'depth_grad_nonzero_when_not_detached': off['max_abs_grad_wrt_depth'] > 1e-6,
        'depth_grad_exactly_zero_when_detached': on['max_abs_grad_wrt_depth'] == 0.0,
        'normal_grad_nonzero_when_not_detached': off['max_abs_grad_wrt_normal'] > 1e-6,
        'normal_grad_nonzero_when_detached': on['max_abs_grad_wrt_normal'] > 1e-6,
        'normal_grad_identical_both_modes':
            abs(off['max_abs_grad_wrt_normal'] - on['max_abs_grad_wrt_normal']) < 1e-9,
        'alias_when_not_detached': off['depth_for_loss_is_depth'] is True,
        'not_alias_when_detached': on['depth_for_loss_is_depth'] is False,
    }
    for k, v in res.items():
        print(f'[{k}] loss={v["loss"]:.8f}  |dL/dD|max={v["max_abs_grad_wrt_depth"]:.6e}  '
              f'|dL/dN|max={v["max_abs_grad_wrt_normal"]:.6e}  valid={v["valid_ratio"]:.4f}')
    n_ok = sum(checks.values())
    for k, v in checks.items():
        print(f'  [{"PASS" if v else "FAIL"}] {k}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'values': res, 'checks': checks, 'n_passed': n_ok, 'n_total': len(checks)},
              open(a.out, 'w'), indent=2)
    print(f'{n_ok}/{len(checks)} checks passed -> {a.out}')
    sys.exit(0 if n_ok == len(checks) else 1)


if __name__ == '__main__':
    main()
