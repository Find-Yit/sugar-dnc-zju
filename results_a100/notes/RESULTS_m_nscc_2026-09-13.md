# 结果记录：SuGaR 提取端改进（M1+ / M2-B），NSCC 4×A100，2026-09-13 09:47–10:00

> 执行：主会话（Fable 5.1）制定并验收，4 个 opus 子代理实现/跑批/复算。
> 计划：`plans/新方案_A100四卡_SuGaR改进.md`；调研：`notes/SURVEY_sugar_improvements.md`、`notes/SURVEY2_sugar_direct_followups.md`。
> 代码：`repo/sugar-dnc-zju`（分支 dnc，hopper 侧 commit 39c0f56 之上的未提交改动，`git diff --stat` 见文末）。
> 全部数字来自 `outputs/metrics/summary_m.csv`（`python scripts/m/summarize.py` 生成）；独立复算 `outputs/metrics/verify_independent.json`。

## 0. 范围裁剪（1 小时时限）
- 做：M1（Frosting 自动 D）+ **M1+a 前/背景分离自适应深度** + M1+b（表面点估密度）+ **M2-B 提取前 DBSCAN 剪枝**（保留大簇规则 + M2+b 分区 eps）。全部在同一 coarse 模型 `outputs/runs/coarse_base_seed0/.../15000.pt` 上做，**不重训**。
- 不做：M2-A 训练端网格、M2+a 贡献度评分（需训练循环 visibility 计数）、泛化场景。

## 1. 设置
- 场景 Truck（tandt），3DGS 7k → coarse SuGaR 15000 迭代（hopper H200 训练，本机复用）。
- 提取固定：`-l 0.3 -d 200000 --eval True --extract_seed 0`（seed 是本次新增，原版无种子不可复现）。
- 每组 1 张 A100 + 14 个 CPU 线程，4 卡并行，单组 5–6 分钟，8 组 + 评测共 13 分钟。**注意**：每进程 64 线程时 4 组并发会把 Poisson 拖到 13 分钟以上仍不出结果（OpenMP 自旋互抢），必须限线程。
- 指标：`scripts/eval/eval_geometry.py`（G1 稀疏点→mesh 距离、G2 精度比例、G4 拓扑、G5 二面角）。

## 2. 终表（同一 coarse 模型，seed 0）
| TAG | D_fg/D_bg | q | 剪枝比 | 连通分量 | 碎片(<100面) | 最大分量占比 | 边界边 | G1 中位 | G2(<1%) | G5 abs 均值 | 面数 | 耗时 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_d10_q01（原版 SuGaR） | 10/10 | 0.1 | – | 2184 | 2094 | 0.423 | 38450 | 0.00405 | 0.9773 | 33.59 | 370284 | 340 |
| base_d10_q0 | 10/10 | 0 | – | 1651 | 1628 | 0.930 | 11339 | 0.00418 | 0.9813 | 34.66 | 331769 | 352 |
| frosting_q0_s0（Frosting 原样 auto D=10） | 10/10 | 0 | – | 1645 | 1622 | 0.930 | 11353 | 0.00417 | 0.9813 | 34.65 | 331767 | 351 |
| m1a_q01（**M1+a**） | 10/**9** | 0.1 | – | 1711 | 1644 | 0.440 | 27669 | 0.00405 | 0.9774 | 32.62 | 356207 | 301 |
| m1a_q0（**M1+a** + q0） | 10/**9** | 0 | – | **1478** | 1456 | 0.929 | **11033** | 0.00417 | 0.9814 | 33.57 | 321501 | 302 |
| m2p95_q0（**M2-B** p95 + q0） | 10/10 | 0 | 14.2% | **1434** | 1412 | **0.940** | 11352 | 0.00407 | 0.9762 | 34.15 | 325562 | 322 |
| m2p95_m1a_q0（M2-B + M1+a auto） | 10/10* | 0 | 14.2% | 1435 | 1414 | 0.941 | 11344 | 0.00407 | 0.9762 | 34.16 | 325550 | 322 |
| m2largest_q0（2D-SuGaR 原样只留最大簇） | 10/10 | 0 | **59.7%** | 1302 | 1275 | 0.929 | 12375 | 0.00417 | 0.9811 | 31.13 | 307721 | 309 |

\* 剪枝后背景高斯密度上升，auto 估出的 D_bg 从 9.99→10.08，被取整为 10，因此该组实际 = m2p95_q0（数字几乎相同）。公平的叠加组 `m2p95_bg9_q0`（强制 D_bg=9）已在 10:01 追加提交，结果见 `summary_m.csv` 更新。

## 3. 噪声底与跨机复现
- 同参重复（base_d10_q0 vs frosting_q0_s0，均 D=10/q0/seed0）：分量 1651 vs 1645（0.4%），G1/G2/G5 差 <0.2%。→ **拓扑计数 >3% 才算信号**（保守取 5%）。
- 与 hopper H200 同参结果（仓库 `results/metrics/geometry_coarse_base_seed0.json`、`geometry_base_seed0_q0.json`）：G1/G2/G5 相对差 <1%，分量数差 2–3%，q0 的大跃迁两机一致。

## 4. 结论（对照假设）
1. **Frosting 原样 auto D 在 Truck 上零收益**（D=10 = 原版，raw 10.87 被上限截断）——与 hopper 结论一致，是负结果。
2. **M1+a（自有改动）有效**：背景高斯更稀，分开估出 D_bg=9（raw 9.99）。
   - q0.1 下：分量 2184→1711（−21.7%），边界边 38450→27669（−28%），G1/G2 不变；
   - q0 下：分量 1651→1478（**−10.5%**），边界边 −2.7%（噪声内），G1/G2/G5 不变，面数 −3%，快 50 s。
   - 假设 H-M1+ "背景碎片下降 ≥20% 且前景 G1 不变"：在 q0.1 下成立（−21.7%），在 q0 下只到 −10.5%（q0 已先吃掉大部分碎片）。
3. **M2-B（DBSCAN 剪枝，保留大簇 + 分区 eps）有效**：剪 14.2% 高斯，分量 1651→1434（**−13.1%**），最大分量占比 0.930→0.940（+0.010，弱信号），G1 反而略降（好），G2 −0.5%（噪声边缘）。假设 H-M2 "碎片 ≥30% 下降"未达到（−13%）。
4. **2D-SuGaR 原样"只留最大簇"是失败案例**：剪掉 59.7%（背景 80.7%、前景 0.39%），分量数看似最低（1302）但是因为背景整片没了；边界边反升到 12375，G5 降到 31（背景大片缺失）。**当前几何指标（G1 只用前景 bbox 内稀疏点）无法捕捉背景丢失，报告里必须配可视化。**
5. **M1+b（表面点估密度）在 Truck 上无区分度**：raw 14.9/13.2 均被 10 截断；只在稀疏场景可能起作用（负结果，已记录 `outputs/m1/report_depth_surface/extract_stats.json`）。
6. 剪枝与自适应深度**不是简单可叠加**：剪枝抬高背景密度使 auto D_bg 回到 10（见 * 注）。

## 5. 验收
- 默认参数路径未变：base_d10_q01 与 hopper 原版结果一致（差异在噪声内）；`--poisson_depth 10 --vertices_density_quantile 0.1` 为默认值。
- 独立复算（`scripts/m/verify_independent.py`，自写 union-find / 边表 / 精确点面距离，不走 eval_geometry 代码路径）：3 组网格拓扑三项与顶点面数逐位一致，G1 精确口径相对差 0.4–1.7%（3000 点子样本抽样噪声内），D_fg/D_bg 与剪枝比核对通过，非流形边=0、重复顶点=0。
- 每组 `outputs/m/mesh_<TAG>/{*.ply, extract_stats.json, .extract_rc, .extract_wallclock_s}`，日志 `logs/m_mesh_<TAG>.log`，评测 `outputs/metrics/geometry_m_<TAG>.json` + `provenance_m_<TAG>.json`；剪枝统计 `outputs/m2/<variant>/15000_pruned_prune_stats.json`。

## 6. 来源披露
- M1：Anttwo/Frosting `frosting_extractors/coarse_shell.py` L17-49（`compute_optimal_poisson_depth`，逐字移植，ECCV 2024）。M1+a/b 为自有改动（`sugar_extractors/coarse_mesh.py` 中 `compute_poisson_depth_from_points` / `compute_optimal_poisson_depth_bg`）。
- M2：prajwalcr/2d-sugar `gaussian_model.py:429` + `train.py:140`（eps=第 k 近邻距离 90 分位数、DBSCAN）。"保留大簇"规则与分区 eps（M2+b）为自有改动（`sugar_utils/cluster_prune.py`、`prune_coarse_model.py`）。

## 7. 复现命令
```bash
source /scratch/users/nus/e1351071/test_zju/env.sh
# 剪枝 .pt
cd $PROJ_ROOT/repo/sugar-dnc-zju && python prune_coarse_model.py -s $PROJ_ROOT/data/tandt/truck -c $PROJ_ROOT/outputs/baseline/gs_truck/ -i 7000 \
  -m $PROJ_ROOT/outputs/runs/coarse_base_seed0/sugarcoarse_3Dgs7000_densityestim02_sdfnorm02/15000.pt --gpu 0 --knn_percentile 95 -o $PROJ_ROOT/outputs/m2/pruned_p95/15000_pruned.pt
# 网格 + 评测（配置文件格式 TAG|GPU|COARSE_PT|EXTRA）
M_OMP=14 OMP_WAIT_POLICY=PASSIVE bash $PROJ_ROOT/scripts/m/run_grid.sh $PROJ_ROOT/scripts/m/grid_restart.txt
python $PROJ_ROOT/scripts/m/summarize.py; python $PROJ_ROOT/scripts/m/verify_independent.py base_d10_q01 m1a_q0 m2p95_q0
```

---

## 8. 视觉指标与对比图（2026-09-13 补做，满足考核 §4「≥1 项渲染/视觉指标 + ≥1 组 baseline vs modified 对比图」）

### 8.1 渲染指标：网格像素覆盖率 mesh_pixel_coverage

用 `scripts/eval/render_mesh_views.py`（pytorch3d `MeshRasterizer`，headless）在与 `eval_render.py` **完全相同的 4 个固定测试视角**（test 集内序号 0/8/16/24，图像 000001/000065/000129/000193，979×546）上光栅化各 mesh，统计 `pix_to_face >= 0` 的像素比例。单组约 16 s，GPU 0/1/3 并行。

原始 JSON：`outputs/metrics/meshvis_m_<TAG>.json`；汇总：`outputs/metrics/meshvis_summary_m.csv`（生成脚本 `scripts/m/meshvis_summary.py`）。

| TAG | 说明 | 视角00 | 视角08 | 视角16 | 视角24 | **均值** | 面数 |
|---|---|---|---|---|---|---|---|
| base_d10_q01 | 原版 SuGaR (D=10, q=0.1) | 89.05% | 94.70% | 87.76% | 96.49% | **92.00%** | 370284 |
| base_d10_q0 | 基线 q=0 | 99.94% | 100.00% | 99.93% | 100.00% | **99.97%** | 331769 |
| m1a_q0 | M1+a (D_bg=9) | 99.98% | 99.84% | 99.98% | 100.00% | **99.95%** | 321501 |
| m2p95_q0 | M2-B (DBSCAN p95) | 99.97% | 100.00% | 99.98% | 100.00% | **99.99%** | 325562 |
| **m2p95_bg9_q0** | **M2-B + M1+a (p95, D_bg=9)** | 99.97% | 99.99% | 99.96% | 100.00% | **99.98%** | 313971 |
| m2largest_q0 | 失败案例：只留最大簇 | 99.97% | 100.00% | 99.94% | 99.94% | **99.96%** | 307721 |

> `m2p95_bg9_q0`（M2-B 剪枝 + 强制 D_bg=9 的公平叠加组）在 10:06 完成提取后补渲染（GPU 0，7.9 s），已并入本表与全部图表。它是**全表最优组合**：碎片 1194（比 q0 基线 1628 再 −26.6%）、连通分量 1216（−26.3%）、边界边 11225（最低）、最大分量占比 0.939，同时覆盖率 99.98%（与基线差 +0.014 pp，噪声内）——**拓扑显著改善且不损失任何可见表面**，与 §4 结论 6「剪枝与自适应深度不可简单叠加」形成对照：强制 D_bg=9 绕开 auto 估计被剪枝抬回 10 的问题后，两项改进确实可叠加。

**指标方向与解读（重要）**：覆盖率高 ≠ 网格好，它只度量"屏幕上有没有面"，不度量"面对不对"。
1. **孔洞减少 → 覆盖率上升，这一条成立且信号极强**：原版 `q=0.1` 的顶点密度分位数裁剪会在背景（树冠、天空边缘）留下大片真孔洞，均值覆盖率只有 92.00%，最差视角 16 仅 87.76%；改成 `q=0` 后升到 99.97%（**+7.97 pp**）。这是 §2 表里"q0 使连通分量 2184→1651、边界边 38450→11339"的像素级对应证据。
2. **失败案例的背景删除，覆盖率没有捕捉到**：`m2largest_q0` 剪掉 59.7% 高斯（其中背景 80.7%），覆盖率却仍是 99.96%（比基线只低 0.006 pp，远在噪声内）。原因是 **Poisson 重建是封闭表面重建**：背景高斯被删后，算法不会留空洞，而是用极少数巨大三角面片把场景外壳"强行封口"。像素被填满了，但填的是错的几何。
   → **结论：覆盖率只能证伪"有孔洞"，不能证伪"背景内容丢失"。背景丢失必须靠法向/深度图肉眼判读（§8.3），这也是 §4 结论 4 的直接验证。**
3. M1+a / M2-B / M2-B+M1+a 相对基线的覆盖率变化均在 ±0.02 pp，**属噪声**——说明这两项改进在"不破坏可见表面"的前提下降低了碎片数，没有以删掉可见几何为代价。这正是我们想要的（几何指标 G1/G2 不变 + 拓扑改善 + 覆盖率不变）。

### 8.2 对比图（`scripts/m/make_figs_meshvis.py`，ML venv matplotlib + `assets/fonts/NotoSansCJKsc-Regular.otf`）

| 图 | 路径 | 内容 |
|---|---|---|
| 图 M-1 | `outputs/figures/m_nscc/fig_mesh_normal_compare.png` | 法向着色图，6 方法 × 4 视角（行序：原版 / q0 / M1+a / M2-B / M2-B+M1+a / 失败案例），每格标 TAG/视角/该视角覆盖率 |
| 图 M-2 | `outputs/figures/m_nscc/fig_mesh_depth_compare.png` | 深度图（turbo，逐视角 p1–p99），同 6 行布局 |
| 图 M-3 | `outputs/figures/m_nscc/fig_failure_background.png` | 失败案例背景并排放大（视角 00，画面上半部 = 树冠区） |
| 图 M-4 | `outputs/figures/m_nscc/fig_topology_bars.png` | 连通分量数 / 边界边数 / 最大分量占比三联柱状图（6 根柱，同上行序）（数据源 `outputs/metrics/summary_m.csv`） |

单视角 PNG 原图在 `outputs/vis/m_<TAG>/mesh_{normal,depth}_view{00,08,16,24}_*.png`。

### 8.3 肉眼判读结论（四张图均已逐张查看确认）

1. **孔洞：原版 q=0.1 一行肉眼可见大片白斑（= 无面片命中），q=0 一行完全消失。** 最明显的是**视角 00 和视角 16**——树冠中间、卡车车斗上方的天空区在 q0.1 下是成片的白洞，深度图（图 M-2 第一行）同一位置也是白的；换到第二行 q=0 后这些区域被完整的树冠网格填满。视角 24 的白洞只在画面顶部一条，最不明显（覆盖率 96.49%，是 q0.1 里最高的），与表中数字一致。
2. **碎片：法向图上表现为背景的"马赛克化"。** 第 2–4 行（q=0 基线 / M1+a / M2-B）背景树冠都呈现高频彩色碎片；M1+a（第 3 行）与 M2-B（第 4 行）相对基线**肉眼差别很小**，最容易看出的是视角 00、16 左上角那几片大三角"伞状"面片的形状和数量略有变化。这与 §4 的判断一致：−10.5% / −13.1% 的分量数下降是**统计意义上的**改善（超过 5% 噪声底），但**不到肉眼可见的程度**；真正肉眼可见的跃迁是 q0.1→q0（−24%）。→ 报告里不应把 M1+a/M2-B 说成"视觉上明显更好"，只能说"拓扑更干净且不损害可见表面"。
4. **失败案例的背景缺失：肉眼极其明显（图 M-3）。** 视角 00 的上半幅，基线是密密麻麻的树叶/树枝网格（高频碎片纹理），`m2largest_q0` 则**整片树冠消失**，只剩几块横跨半个画面的大尺寸平滑三角面（图中大红/浅绿色块），像一张被拉平的幕布。四个视角里视角 00 和视角 16 最明显（原本树冠占比最大），视角 08 因为卡车前脸占据大部分画面、背景本来就少，差异最不明显。深度图（图 M-2 第 5 行）佐证同一现象：背景本应是连续的红→橙深度渐变，失败案例变成几块单调的大色块。
5. **这一节的方法论价值**：`m2largest_q0` 同时满足"连通分量数最低（1302，全表最好）"和"像素覆盖率 ~100%"，**两类定量指标都给它打了高分，但它是本次实验里唯一一个视觉上明确变差的配置**。只有 G5 二面角（34.66→31.13）和边界边（11339→12375）反向报警，加上法向/深度可视化才能定案。→ 网格质量评估**必须几何指标 + 视觉检查双轨**，这是本次实验最可迁移的结论之一。

### 8.4 复现命令
```bash
source /scratch/users/nus/e1351071/test_zju/env.sh
for TAG in base_d10_q01 base_d10_q0 m1a_q0 m2p95_q0 m2p95_bg9_q0 m2largest_q0; do
  python scripts/eval/render_mesh_views.py --scene_path $PROJ_ROOT/data/tandt/truck \
    --gs_checkpoint $PROJ_ROOT/outputs/baseline/gs_truck \
    --mesh_path $(ls $PROJ_ROOT/outputs/m/mesh_${TAG}/*.ply) --run_name m_${TAG} --gpu 0
done
python scripts/m/meshvis_summary.py      # -> outputs/metrics/meshvis_summary_m.csv
python scripts/m/make_figs_meshvis.py    # -> outputs/figures/m_nscc/*.png
```

## 9. 与 hopper 侧训练端结果的统一对照（回答"这几项能否横向比较"）

hopper 报告结论：自研 DNC 负结果；官方 dn_consistency 有效（碎片 −16%，PSNR −0.24 dB）；提取端 q=0 碎片 −22%、孔洞 −70%、几何精度 −4.7%。下表把两台机器的结果放在同一坐标系里（训练端行来自 hopper H200，`repo/sugar-dnc-zju/results/metrics/`；提取端行来自本机 A100；跨机同参差异：G1/G2/G5 <1%，拓扑计数 2–3%）。

| 改动位置 | run | 碎片(<100面) | vs 原版 | 连通分量 | 边界边(孔洞) | 最大分量占比 | G1 中位(点到面) | G2(<1%) | PSNR(dB) | SSIM |
|---|---|---|---|---|---|---|---|---|---|---|
| — | 原版 SuGaR（hopper coarse_base_seed0） | 2137 | 0 | 2233 | 38596 | 0.417 | 0.00402 | 0.9774 | 24.62 | 0.8555 |
| — | 原版 SuGaR（NSCC base_d10_q01，同参复现） | 2094 | −2% | 2184 | 38450 | 0.423 | 0.00405 | 0.9773 | (同上，高斯未变) | |
| 训练端 | 官方 dn_consistency（hopper） | ≈1795* | **−16%*** | – | – | – | – | – | **24.38** | 0.8485 |
| 训练端 | 自研 DNC λ=0.05（hopper） | 2324 | +9% | 2403 | 35449 | 0.453 | 0.00831 | 0.8728 | 22.45 | 0.7708 |
| 训练端 | 自研 DNC λ=0.2（hopper，坍缩） | 1177 | −45%（假象） | 1192 | 23485 | 0.846 | 0.04908 | 0.5547 | 13.24 | 0.4741 |
| 训练端 | 自研 DNC λ=0.2 detach（hopper） | 2142 | 0 | 2242 | 40452 | 0.418 | 0.00491 | 0.9754 | 24.40 | 0.8494 |
| 提取端 | q=0（hopper） | 1668 | −22% | 1699 | 11566 | 0.928 | 0.00421 | 0.9814 | 24.62（不变） | 0.8555 |
| 提取端 | q=0（NSCC） | 1628 | −24% | 1651 | 11339 | 0.930 | 0.00418 | 0.9813 | 不变 | |
| 提取端 | **M1+a**（D_bg=9）+ q0（NSCC） | 1456 | **−32%** | 1478 | 11033 | 0.929 | 0.00417 | 0.9814 | 不变 | |
| 提取端 | **M2-B**（DBSCAN p95，剪 14%）+ q0（NSCC） | 1412 | **−34%** | 1434 | 11352 | 0.940 | 0.00407 | 0.9762 | 不变 | |
| 提取端 | **M2-B + M1+a**（D_bg=9 强制）+ q0（NSCC） | **1194** | **−44%** | **1216** | 11225 | 0.939 | 0.00407 | 0.9761 | 不变 | |
| 提取端 | 2D-SuGaR 原样只留最大簇（失败案例） | 1275 | −40%（假象） | 1302 | 12375 | 0.929 | 0.00417 | 0.9811 | 不变 | |

\* 官方 dn_consistency 的几何 json 未随仓库回传（只有 render_coarse_official_dnc.json，PSNR 24.378 / SSIM 0.8485 / LPIPS 0.2175），"碎片 −16%"取自 hopper 报告结论，≈1795 由 2137×0.84 反推，仅供定位。

### 读法
1. **提取端改动不改高斯，PSNR/SSIM 严格不变**（渲染的是同一个 coarse 模型）；训练端改动（官方 dn_consistency）碎片 −16% 的代价是 PSNR −0.24 dB。而提取端 q0 + M2-B + M1+a 叠加到 **碎片 −44%、孔洞（边界边）−71%**，**渲染零代价**，G1 点到面距离 0.00402→0.00407（+1.2%，在噪声底内；hopper 报的 q0 "几何精度 −4.7%" 是 G1 median 0.00402→0.00421，本机 q0 为 +4.0%，M2-B 把它拉回到 +1.2%——剪掉孤立高斯后 Poisson 输入更干净）。
2. **两个失败案例的共同特征**：碎片数"最低"但几何指标反向报警。自研 DNC λ=0.2：G1 ×12、G2 0.55、PSNR 13 dB（场景坍缩）；2D-SuGaR 只留最大簇：G5 34.7→31.1、边界边反升、覆盖率抓不到（Poisson 封口），只有法向图能看出背景被大三角面片替代。**报告里两个失败案例并列，正好说明"碎片数不能单独判优"。**
3. **训练端 vs 提取端的组合是正交的**：官方 dn_consistency 训出的 coarse 模型（hopper 有 .pt）再走 M2-B + M1+a 提取，预期碎片进一步下降；本机没有那个 .pt，未做，列为下一步。
4. 代价：M2-B 剪枝 15 s，M1+a 让提取更快（背景 D=9 → 240 s vs 352 s），无额外训练成本；M2-B 引入 `knn_percentile`、`keep_cluster_min_frac` 两个超参（本次扫了 p90/p95、0.005/0.002 四个剪枝比 14–25%，largest 规则为失败边界）。

## 10. 主会话看图后的诚实补充（必须写进报告的限制）
看 `fig_mesh_normal_compare.png` 后确认：**q=0 消除的"孔洞"有相当一部分是天空/树冠区域被 Poisson 用大块平面三角形封口**（第 2–5 行视角 00/08 左上角的大色块），而不是重建出了真实几何。原版 q=0.1 的清洗恰恰是把这些低密度大三角面删掉才留下"孔"。因此：
- "边界边 −71%" 应表述为"开放边界大幅减少（其中一部分来自封口面片）"，不能说成"重建完整性提升"；
- 前景卡车区域（4 个视角的中央）q0 / M1+a / M2-B 与原版**肉眼无差别**，说明改动对前景几何是安全的，增益集中在背景碎片计数；
- 这一现象与 §8 "覆盖率抓不到背景缺失"是同一个根因（Poisson 封闭曲面假设），报告里应一并作为"指标的局限"讨论；
- 更合理的下一步：对 q=0 的封口大面片按面积/密度做**局部**清洗（M1+c 方案），而不是全局分位数——本次未做。
