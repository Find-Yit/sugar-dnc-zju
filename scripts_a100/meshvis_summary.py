#!/usr/bin/env python
"""汇总 meshvis_m_*.json 的 mesh_pixel_coverage -> outputs/metrics/meshvis_summary_m.csv"""
import csv, json, os
P = os.environ.get("PROJ_ROOT", "/scratch/users/nus/e1351071/test_zju")
ORDER = [("base_d10_q01", "原版 SuGaR (d10,q0.1)"), ("base_d10_q0", "基线 q0"),
         ("m1a_q0", "M1+a (D_bg=9)"), ("m2p95_q0", "M2-B (DBSCAN p95)"),
         ("m2p95_bg9_q0", "M2-B+M1a (bg9)"), ("m2largest_q0", "失败案例: 只留最大簇")]
rows, views = [], None
for tag, label in ORDER:
    p = os.path.join(P, "outputs/metrics", f"meshvis_m_{tag}.json")
    if not os.path.isfile(p):
        continue
    d = json.load(open(p))
    cov = {int(k): v for k, v in d["mesh_pixel_coverage"].items()}
    views = sorted(cov) if views is None else views
    names = {int(k): v for k, v in d["image_names"].items()}
    r = {"TAG": tag, "label": label}
    for v in views:
        r[f"cov_view{v:02d}_{names[v]}"] = round(cov[v], 6)
    r["cov_mean"] = round(sum(cov[v] for v in views) / len(views), 6)
    r["n_faces"] = d["config"]["n_faces"]
    rows.append(r)
out = os.path.join(P, "outputs/metrics/meshvis_summary_m.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("CSV ->", out, "\n")
cols = list(rows[0].keys())
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
base = rows[1]["cov_mean"] if len(rows) > 1 else None
for r in rows:
    print("| " + " | ".join(f"{r[c]*100:.2f}%" if c.startswith("cov") else str(r[c]) for c in cols) + " |")
print("\n相对 base_d10_q0 的均值覆盖率变化：")
for r in rows:
    print(f"  {r['TAG']:16s} {(r['cov_mean']-base)*100:+.3f} pp")
