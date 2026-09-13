"""把 outputs/metrics/s34_collect.json 渲染成 SELF_CHECK 的 C3/C4 表格（markdown）。
所有数字直接来自 JSON，杜绝手抄错误。用法: python scripts/render_s34_selfcheck.py > /tmp/tables.md
"""
import json, os

PROJ = "/scratch/e1351071/zju_test"
J = json.load(open(os.path.join(PROJ, "outputs", "metrics", "s34_collect.json")))
R = J["runs"]
ORDER = ["coarse_base_seed0", "coarse_dnc005", "coarse_dnc02"]
LBL = {"coarse_base_seed0": "coarse_base_seed0 (λ=0)",
       "coarse_dnc005": "coarse_dnc005 (λ=0.05)",
       "coarse_dnc02": "coarse_dnc02 (λ=0.2)"}


def rel(p):
    return p.replace(PROJ + "/", "") if p else "—"


print("### C3-a 三组 `.pt` / `.ply` 齐全性\n")
print("| run | λ | GPU | `15000.pt` | 大小 (B) | `.ply` | 大小 (B) |")
print("|---|---|---|---|---|---|---|")
def num(v):
    return f"{v:,}" if isinstance(v, int) else "—"


for r in ORDER:
    d = R[r]; st = d["train_stats"] or {}
    mb = d["mesh"]["file_bytes"] if d.get("mesh") else None
    print(f"| `{r}` | {d['lambda']} | {st.get('gpu_index')} | `{rel(d['pt_path'])}` | {num(d['pt_bytes'])} "
          f"| `{rel(d['ply_path'])}` | {num(mb)} |")

print("\n### C3-b `train_stats.json` 关键字段\n")
rows = [
    ("训练墙钟 (min)", "train_wallclock_min", "{:.2f}"),
    ("总墙钟含加载 (min)", "total_wallclock_min", "{:.2f}"),
    ("每迭代耗时 (ms/iter)", "mean_time_per_iteration_ms", "{:.1f}"),
    ("峰值显存 allocated (MiB)", "max_memory_allocated_MiB", "{:.1f}"),
    ("峰值显存 reserved (MiB)", "max_memory_reserved_MiB", "{:.1f}"),
    ("15k 高斯数", "n_gaussians_final", "{:,}"),
    ("完成迭代数", "n_iterations_done", "{:,}"),
    ("最终 loss", "final_loss", "{:.7f}"),
    ("末次 L_dnc", "last_l_dnc", "{}"),
    ("末次有效像素比", "last_dnc_valid_pixel_ratio", "{}"),
]
print("| 字段 | " + " | ".join(LBL[r] for r in ORDER) + " |")
print("|---|" + "---|" * len(ORDER))
for label, key, fmt in rows:
    cells = []
    for r in ORDER:
        v = (R[r]["train_stats"] or {}).get(key)
        if v is None:
            cells.append("—")
        elif fmt == "{}":
            cells.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        else:
            cells.append(fmt.format(v))
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n### C3-c λ 组 `dnc_log.csv`：L_dnc 走势 + sdf_better_normal_loss（H3）\n")
print("| 量 | coarse_dnc005 (λ=0.05) | coarse_dnc02 (λ=0.2) |")
print("|---|---|---|")


def g(run, *path):
    d = R[run].get("dnc_log")
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def fmt2(run, *path, f="{:.6f}"):
    v = g(run, *path)
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        try:
            return f.format(v)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


LAM = ["coarse_dnc005", "coarse_dnc02"]
metrics = [
    ("CSV 行数", ("n_rows",), "{:d}"),
    ("迭代范围 首", ("iteration_first",), "{:.0f}"),
    ("迭代范围 末", ("iteration_last",), "{:.0f}"),
    ("**L_dnc 首** (iter 首)", ("l_dnc", "first"), "{:.6f}"),
    ("**L_dnc 中**", ("l_dnc", "mid"), "{:.6f}"),
    ("**L_dnc 末**", ("l_dnc", "last"), "{:.6f}"),
    ("L_dnc 首 10% 均值", ("l_dnc_mean_first10pct",), "{:.6f}"),
    ("L_dnc 末 10% 均值", ("l_dnc_mean_last10pct",), "{:.6f}"),
    ("L_dnc min / max", None, None),
    ("L_dnc 线性斜率 (/1000 iter)", ("l_dnc_slope_per_1k_iter",), "{:+.6f}"),
    ("L_dnc 全部有限（无 NaN）", ("l_dnc_all_finite",), "{}"),
    ("sdf_better_normal_loss 首", ("sdf_better_normal_loss", "first"), "{:.6f}"),
    ("sdf_better_normal_loss 中", ("sdf_better_normal_loss", "mid"), "{:.6f}"),
    ("sdf_better_normal_loss 末", ("sdf_better_normal_loss", "last"), "{:.6f}"),
    ("sdf_better_normal 首 10% 均值", ("sdf_better_normal_mean_first10pct",), "{:.6f}"),
    ("sdf_better_normal 末 10% 均值", ("sdf_better_normal_mean_last10pct",), "{:.6f}"),
    ("sdf_better_normal 斜率 (/1000 iter)", ("sdf_better_normal_slope_per_1k_iter",), "{:+.6f}"),
    ("sdf_estimation_loss 首 / 末", None, None),
    ("有效像素比 首 / 中 / 末", None, None),
    ("有效像素比 全程均值", ("valid_pixel_ratio_mean",), "{:.4f}"),
    ("dnc_vis PNG 张数", None, None),
]
for label, path, f in metrics:
    if label.startswith("L_dnc min"):
        cells = [f"{g(r,'l_dnc_min'):.6f} / {g(r,'l_dnc_max'):.6f}" if g(r, 'l_dnc_min') is not None else "—" for r in LAM]
    elif label.startswith("sdf_estimation_loss 首"):
        cells = [f"{g(r,'sdf_estimation_loss','first'):.6f} / {g(r,'sdf_estimation_loss','last'):.6f}" if g(r, 'sdf_estimation_loss') else "—" for r in LAM]
    elif label.startswith("有效像素比 首"):
        cells = [f"{g(r,'valid_pixel_ratio','first'):.4f} / {g(r,'valid_pixel_ratio','mid'):.4f} / {g(r,'valid_pixel_ratio','last'):.4f}" if g(r, 'valid_pixel_ratio') else "—" for r in LAM]
    elif label.startswith("dnc_vis"):
        cells = [str(R[r]["n_dnc_vis_png"]) for r in LAM]
    else:
        cells = [fmt2(r, *path, f=f) for r in LAM]
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\n### C3-d 三个 `.ply` 的 open3d 独立核验\n")
print("| run | 顶点数 | 面数 | 顶点全有限 | any NaN 顶点 | 顶点色 NaN | edge manifold | 连通分量数 | 最大分量面数占比 | <100 面碎片数 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in ORDER:
    m = R[r]["mesh"]
    if not m:
        print(f"| `{r}` | — | — | — | — | — | — | — | — | — |")
        continue
    print(f"| `{r}` | {m['n_vertices']:,} | {m['n_triangles']:,} | {m['vertices_finite']} | {m['any_nan_vertices']} "
          f"| {m.get('any_nan_vertex_colors')} | {m['edge_manifold']} | {m['n_connected_components']:,} "
          f"| {m['largest_component_frac']*100:.2f}% | {m['n_components_lt100_tri']:,} |")

print("\n### C4 三组是否同 ckpt / seed / 提取参数\n")
c = J.get("consistency_C4", {})
print("| 检查项 | 值 | 一致 |")
print("|---|---|---|")
print(f"| 3DGS checkpoint | `{c.get('gs_checkpoint')}` | {c.get('same_gs_checkpoint')} |")
print(f"| 场景路径 | `{c.get('scene')}` | {c.get('same_scene')} |")
print(f"| seed | {c.get('seed')} | {c.get('same_seed')} |")
print(f"| `-i` 载入迭代 | {(R[ORDER[0]]['train_stats'] or {}).get('iteration_to_load')} | {c.get('same_iteration_to_load')} |")
print(f"| 总迭代 num_iterations | {(R[ORDER[0]]['train_stats'] or {}).get('num_iterations')} | {c.get('same_num_iterations')} |")
print(f"| 实际完成迭代数 | {c.get('n_iterations_done')} | {c.get('same_n_iterations_done')} |")
print(f"| SDF 系数 (estim, normal) | {(R[ORDER[0]]['train_stats'] or {}).get('sdf_estimation_factor')}, "
      f"{(R[ORDER[0]]['train_stats'] or {}).get('sdf_better_normal_factor')} | {c.get('same_sdf_factors')} |")
print(f"| eval 划分 | {(R[ORDER[0]]['train_stats'] or {}).get('eval_split')} | {c.get('same_eval_split')} |")
print(f"| dnc_factor（**故意不同**） | {c.get('dnc_factors')} | — |")
print(f"| dnc_start | {c.get('dnc_starts')} | {len(set(c.get('dnc_starts') or []))==1} |")
