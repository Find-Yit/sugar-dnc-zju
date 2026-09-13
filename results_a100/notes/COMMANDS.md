# COMMANDS.md — 每组 run 的确切命令、日志、起止时间与耗时（清单第 4 项）

> 命令行**原样摘自** `logs/m_mesh_<TAG>.log` 头部的 `command :` 行；
> 起止时间摘自同一日志的 `=== M MESH START/END ===` 行；
> 耗时以 `outputs/m/mesh_<TAG>/.extract_wallclock_s`（= `summary.csv:extract_wallclock_s`）为准，
> 与日志 END 行的 `wallclock=` 相差 5–8 s（后者含脚本外壳与进程启停）。
>
> 共同前缀（所有提取命令）：`cd $PROJ_ROOT/repo/sugar-dnc-zju`，先 `source $PROJ_ROOT/env.sh`；
> `$PROJ_ROOT = /scratch/users/nus/e1351071/test_zju`。
> 注：日志里 `--poisson_depth` / `--vertices_density_quantile` 出现**两次**是运行脚本先写默认值再追加 EXTRA 覆盖，
> **后出现的值生效**（argparse 后者覆盖前者），与 `extract_stats.json` 里的 `*_used` 字段一致。

## 0. 总览表

| TAG | 说明 | 日志 | START | END | 日志 wallclock | `.extract_wallclock_s` |
|---|---|---|---|---|---|---|
| `base_d10_q01` | **原版 SuGaR 默认**（D=10/10, q=0.1） | `logs/m_mesh_base_d10_q01.log` | 09:47:18 | 09:53:05 | 347 s | **340.06** |
| `base_d10_q0` | q=0 基线 | `logs/m_mesh_base_d10_q0.log` | 09:47:18 | 09:53:17 | 359 s | **351.66** |
| `frosting_q0_s0` | Frosting 原样 auto D | `logs/m_mesh_frosting_q0_s0.log` | 09:53:26 | 09:59:22 | 356 s | **351.12** |
| `m1a_q0` | **M1+a**（auto D_fg/D_bg → 10/9）+ q0 | `logs/m_mesh_m1a_q0.log` | 09:47:18 | 09:52:27 | 309 s | **301.71** |
| `m1a_q01` | M1+a + q0.1 | `logs/m_mesh_m1a_q01.log` | 09:52:56 | 09:58:01 | 305 s | **301.02** |
| `m2p95_q0` | **M2-B**（DBSCAN p95 剪枝）+ q0 | `logs/m_mesh_m2p95_q0.log` | 09:47:19 | 09:52:47 | 328 s | **321.87** |
| `m2p95_m1a_q0` | M2-B + M1+a auto（auto 回落到 D_bg=10） | `logs/m_mesh_m2p95_m1a_q0.log` | 09:52:35 | 09:58:02 | 327 s | **322.12** |
| **`m2p95_bg9_q0`** | **最优组合**：M2-B + 强制 D_bg=9 + q0 | `logs/m_mesh_m2p95_bg9_q0.log` | **10:00:16** | **10:04:22** | 246 s | **240.16** |
| `m2largest_q0` | 失败案例：只留最大簇 | `logs/m_mesh_m2largest_q0.log` | 09:53:16 | 09:58:25 | 309 s | **303.98** |

> 报告中「原版 352 s vs 最优 240 s」的出处：
> 352 s = `base_d10_q0` 的 351.66 s（q=0 基线；原版默认 `base_d10_q01` 为 340.06 s）；
> 240 s = `m2p95_bg9_q0` 的 240.16 s。加速来自剪掉 14.2% 高斯 + 背景 Poisson 深度 10→9。
> 日期均为 **2026-09-13**（时区 Asia）。

## 1. 逐组完整命令行

### base_d10_q01（原版 SuGaR 默认参数）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt -l 0.3 -d 200000 --eval True --gpu 1 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_base_d10_q01 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth 10 --vertices_density_quantile 0.1 --extract_seed 0
```

### base_d10_q0
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt -l 0.3 -d 200000 --eval True --gpu 0 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_base_d10_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth 10 --vertices_density_quantile 0 --extract_seed 0
```

### frosting_q0_s0（Frosting 原样 auto D）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt -l 0.3 -d 200000 --eval True --gpu 0 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_frosting_q0_s0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth auto --vertices_density_quantile 0 --extract_seed 0
```

### m1a_q0（M1+a）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt -l 0.3 -d 200000 --eval True --gpu 2 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m1a_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth auto --poisson_depth_bg auto --depth_estimate_source centers --vertices_density_quantile 0 --extract_seed 0
```

### m1a_q01（M1+a + q0.1）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt -l 0.3 -d 200000 --eval True --gpu 3 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m1a_q01 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth auto --poisson_depth_bg auto --depth_estimate_source centers --vertices_density_quantile 0.1 --extract_seed 0
```

### m2p95_q0（M2-B）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/m2/pruned_p95/15000_pruned.pt -l 0.3 -d 200000 --eval True --gpu 3 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m2p95_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth 10 --vertices_density_quantile 0 --extract_seed 0
```

### m2p95_m1a_q0（M2-B + M1+a auto）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/m2/pruned_p95/15000_pruned.pt -l 0.3 -d 200000 --eval True --gpu 2 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m2p95_m1a_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth auto --poisson_depth_bg auto --depth_estimate_source centers --vertices_density_quantile 0 --extract_seed 0
```

### m2p95_bg9_q0（**最优组合**，240 s）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/m2/pruned_p95/15000_pruned.pt -l 0.3 -d 200000 --eval True --gpu 2 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m2p95_bg9_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth 10 --poisson_depth_bg 9 --vertices_density_quantile 0 --extract_seed 0
```

### m2largest_q0（失败案例）
```
python extract_mesh.py -s /scratch/users/nus/e1351071/test_zju/data/tandt/truck -c /scratch/users/nus/e1351071/test_zju/outputs/baseline/gs_truck/ -i 7000 -m /scratch/users/nus/e1351071/test_zju/outputs/m2/pruned_largest_global/15000_pruned.pt -l 0.3 -d 200000 --eval True --gpu 1 -o /scratch/users/nus/e1351071/test_zju/outputs/m/mesh_m2largest_q0 --poisson_depth 10 --vertices_density_quantile 0.1 --cell_size_nn_distance_ratio 100 --poisson_depth 10 --vertices_density_quantile 0 --extract_seed 0
```

## 2. M2-B 剪枝命令（生成 `outputs/m2/<variant>/15000_pruned.pt`）

参数取自各变体的 `metrics/prune_stats_<variant>.json`（字段 `knn_percentile`、`keep_cluster_min_frac`、
`rule`、`separate_fg_bg`、`min_samples`、`gpu_index`、`output_path`）。共同前缀：
`cd $PROJ_ROOT/repo/sugar-dnc-zju`，`M = .../outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt`。

| variant | timestamp | 命令 |
|---|---|---|
| `default` | 09:25:38 | `python prune_coarse_model.py -s $PROJ_ROOT/data/tandt/truck -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 -m $M --gpu 2 --knn_percentile 90 --keep_cluster_min_frac 0.005 -o $PROJ_ROOT/outputs/m2/pruned_default/15000_pruned.pt` |
| **`p95`**（用于最优组合） | 09:26:56 | `python prune_coarse_model.py -s $PROJ_ROOT/data/tandt/truck -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 -m $M --gpu 2 --knn_percentile 95 --keep_cluster_min_frac 0.005 -o $PROJ_ROOT/outputs/m2/pruned_p95/15000_pruned.pt` |
| `frac002` | 09:27:17 | `python prune_coarse_model.py -s $PROJ_ROOT/data/tandt/truck -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 -m $M --gpu 2 --knn_percentile 90 --keep_cluster_min_frac 0.002 -o $PROJ_ROOT/outputs/m2/pruned_frac002/15000_pruned.pt` |
| `largest_global` | 09:27:44 | `python prune_coarse_model.py -s $PROJ_ROOT/data/tandt/truck -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 -m $M --gpu 2 --prune_rule largest --no_separate_fg_bg -o $PROJ_ROOT/outputs/m2/pruned_largest_global/15000_pruned.pt` |

剪枝耗时（`prune_seconds` / `total_seconds_script`）：default 5.68 / 9.64 s、p95 **4.39 / 9.80 s**、
frac002 7.41 / 13.43 s、largest_global 12.77 / 16.75 s。均为 CPU DBSCAN（sklearn），一次性成本。

> **[未提供]**：`prune_coarse_model.py` 的原始调用日志（`logs/` 下无 `*prune*.log`，
> 当时在交互 shell 中直接运行）。上表命令由各 `prune_stats_*.json` 中记录的实际参数值反推，
> 参数与 `RESULTS_m_nscc_2026-09-13.md` §7 给出的复现命令一致；JSON 中的
> `timestamp`、`coarse_model_path`、`output_path`、`gpu_index` 为运行时直接写入，可信。

## 3. 网格 + 评测的批量驱动
```bash
source /scratch/users/nus/e1351071/test_zju/env.sh
M_OMP=14 OMP_WAIT_POLICY=PASSIVE bash $PROJ_ROOT/scripts/m/run_grid.sh $PROJ_ROOT/scripts/m/grid_restart.txt
python $PROJ_ROOT/scripts/m/summarize.py          # -> outputs/metrics/summary_m.csv
python $PROJ_ROOT/scripts/m/verify_independent.py base_d10_q01 m1a_q0 m2p95_q0
```
配置文件格式 `TAG|GPU|COARSE_PT|EXTRA`；日志 `logs/m_grid.log`，评测日志 `logs/m_eval_<TAG>.log`。
**`M_OMP=14` 是必需的**：每进程 64 线程时 4 组并发会 OpenMP 自旋互抢，Poisson 13 min 仍不出结果
（首次尝试即因此中止，见 `logs/m_grid_aborted.log`、`logs/m_mesh_*_aborted.log`）。
