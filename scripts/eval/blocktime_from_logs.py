#!/usr/bin/env python
"""S3.e 附属工具：从各 coarse 训练日志里重算「每迭代耗时」，并剔除被 GPU 争用污染的分块。

为什么需要它
------------
`train_stats.json` 里的 `mean_time_per_iteration_ms` 是整段 7001–15000 的平均，混进了
7001–9000 这段「还没开启任何 DNC」的便宜迭代；而且 `coarse_official_dnc` 的最后约 400 迭代
与别人后起的 `extract_mesh.py` 抢了同一张 H200。SuGaR 的训练器本来就每 200 迭代打印一次
`computed in X minutes`，本脚本把这些分块耗时解析出来：

  - 分块 0        = 第 7000 迭代单步（可忽略）
  - 分块 1..10    = 7001..9000，DNC 尚未生效
  - 分块 11..40   = 9001..15000，DNC 生效段  <- 真正要比的就是这一段
  - 争用判定      = 该分块 ms/iter > 1.5 x (DNC 生效段中位数) 则剔除，并在输出里列出被剔除的值

求和法的精度已用 coarse_dnc005 校验：分块求和 13.9179 min vs 其 train_stats.json 的
train_wallclock_min 13.93 min，偏差 < 0.1%。

用法:
    python scripts/eval/blocktime_from_logs.py
    python scripts/eval/blocktime_from_logs.py --out outputs/metrics/blocktime_dnc_active.json
"""
import argparse
import json
import os
import re
import statistics

PROJ_ROOT = os.environ.get("PROJ_ROOT", "/scratch/e1351071/zju_test")
DEFAULT_RUNS = [
    "coarse_base_seed0",
    "coarse_dnc005",
    "coarse_dnc02",
    "coarse_dnc02_detach",
    "coarse_official_dnc",
]


def blocks_ms_per_iter(log_path, iters_per_block=200):
    """每个 200-迭代分块的 ms/iter（已跳过第一个单步分块）。"""
    secs = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = re.search(r"computed in ([0-9.eE+-]+) minutes", line)
            if m:
                secs.append(float(m.group(1)) * 60.0)
    if not secs:
        raise SystemExit(f"[ERROR] {log_path} 里没有 'computed in ... minutes' 行")
    return [s / iters_per_block * 1000.0 for s in secs[1:]], secs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=str, default=",".join(DEFAULT_RUNS))
    p.add_argument("--log_tmpl", type=str, default=os.path.join(PROJ_ROOT, "logs", "s34_{run}.log"))
    p.add_argument("--out", type=str,
                   default=os.path.join(PROJ_ROOT, "outputs", "metrics", "blocktime_dnc_active.json"))
    p.add_argument("--dnc_start_block", type=int, default=10,
                   help="第几个分块起算作 DNC 生效段（默认 10 = 迭代 9001 起）")
    p.add_argument("--contention_factor", type=float, default=1.5,
                   help="超过 DNC 段中位数的这个倍数即判为被争用，剔除")
    a = p.parse_args()

    result = {"config": {"dnc_start_block": a.dnc_start_block,
                         "contention_factor": a.contention_factor,
                         "iters_per_block": 200,
                         "note": "ms/iter 由训练器自己打印的 'computed in X minutes' 分块耗时换算"},
              "runs": {}}
    base_clean = None
    for run in a.runs.split(","):
        log = a.log_tmpl.format(run=run)
        if not os.path.exists(log):
            print(f"[WARN] 跳过 {run}: 找不到 {log}")
            continue
        per, secs = blocks_ms_per_iter(log)
        pre, post = per[:a.dnc_start_block], per[a.dnc_start_block:]
        med = statistics.median(post)
        clean = [x for x in post if x <= a.contention_factor * med]
        dropped = [x for x in post if x > a.contention_factor * med]
        entry = {
            "log": os.path.relpath(log, PROJ_ROOT),
            "n_blocks": len(per),
            "loop_only_min_sum": sum(secs) / 60.0,
            "pre_dnc_mean_ms_per_iter": sum(pre) / len(pre),
            "dnc_active_mean_ms_per_iter": sum(post) / len(post),
            "dnc_active_median_ms_per_iter": med,
            "dnc_active_clean_mean_ms_per_iter": sum(clean) / len(clean),
            "n_clean_blocks": len(clean),
            "n_dropped_blocks": len(dropped),
            "dropped_ms_per_iter": dropped,
        }
        result["runs"][run] = entry
        if run == "coarse_base_seed0":
            base_clean = entry["dnc_active_clean_mean_ms_per_iter"]

    if base_clean:
        for run, e in result["runs"].items():
            e["rel_to_base_pct"] = 100.0 * (e["dnc_active_clean_mean_ms_per_iter"] - base_clean) / base_clean

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'run':24s} {'7001-9000':>10s} {'9001-15000':>11s} {'clean':>8s} {'rel%':>7s}  dropped")
    for run, e in result["runs"].items():
        print(f"{run:24s} {e['pre_dnc_mean_ms_per_iter']:10.1f} {e['dnc_active_mean_ms_per_iter']:11.1f} "
              f"{e['dnc_active_clean_mean_ms_per_iter']:8.1f} {e.get('rel_to_base_pct', float('nan')):+7.1f}  "
              f"{[round(x) for x in e['dropped_ms_per_iter']]}")
    print(f"\n[OK] 写入 {a.out}")


if __name__ == "__main__":
    main()
