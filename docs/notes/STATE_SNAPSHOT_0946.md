# 状态快照 2026-09-13 09:46（GPU 作业即将到期，主动暂停）

## 已完成（全部产物在 /scratch，作业结束不丢）
- 环境（torch 2.4.1+cu121、pytorch3d 0.7.8、CUDA 扩展）：`env.sh`，`notes/ENV.md`
- 数据 Truck（251 张，219/32 划分）：`data/tandt/truck`
- 3DGS 7k：`outputs/baseline/gs_truck/`（test PSNR 23.91）
- 训练侧对照 6 组（coarse .pt + mesh .ply + 评测）：coarse_baseline（留档）、coarse_base_seed0（λ=0）、coarse_dnc005、coarse_dnc02、coarse_dnc02_detach、coarse_official_dnc → `outputs/runs/`、`outputs/metrics/summary.csv`
- 提取端扫参（同一 base_seed0 模型）：mesh_base_seed0_{pdauto, pd8, q0, pdauto_q0}（几何评测已进 summary.csv；执行者 9 的 SELF_CHECK_阶段3f_poisson.md 可能未写完）
- 报告分析文字：`notes/REPORT_ANALYSIS_fable.md`（P1–P12 定稿；P13 Frosting 一节未写）
- 报告骨架与图表脚本：`scripts/deliverables/make_figures.py`、`make_report.py`；旧版 `outputs/reports/report.pdf`（20 页，占位符未填）；执行者 5 正在生成终稿时被停止，`outputs/reports/` 下可能有半成品
- 新方案 v2.1：`plans/新方案_A100四卡_SuGaR改进.md`；调研：`notes/SURVEY_sugar_improvements.md`、`notes/SURVEY2_sugar_direct_followups.md`
- GitHub：分支 dnc 已推送中期快照（commit 39c0f56）；本次快照再推一次

## 关键结论（写报告用）
- 自研 DNC：λ=0.05 PSNR −2.17 dB 且几何变差；λ=0.2 场景坍缩；detach 深度梯度后 PSNR 回到 −0.22 dB 但几何无改善。
- 官方 dn_consistency（λ=0.05）：碎片 −16.4%、二面角 −10.1%、PSNR −0.24 dB → 思路有效，自研实现细节有缺陷（绝对值 / 法向合成方式 / 掩码，待消融）。
- 提取端：Frosting 自动 D 在 Truck 上几乎无变化（2267 vs 2233 分量）；顶点密度清洗分位数 0.1→0 使碎片 −24%（2137→1668），G1 中位距离 +4%。

## 恢复步骤（下一个作业）
1. `source env.sh`；不需要 GPU 也能完成剩余工作（报告生成只用 CPU + tools/doc_env）。
2. 补写 `notes/REPORT_ANALYSIS_fable.md` 的 P13（Frosting 自动 D + quantile 扫参分析，数字取 summary.csv 的 base_seed0_* 行）。
3. 用 SendMessage 恢复执行者 5（报告终稿）或新派执行者：按 09:32 给执行者 5 的指令生成 report.pdf、fig1–fig7、validate、逐页预览、SELF_CHECK_阶段5_终稿.md。
4. Fable 验收：validate 结果、联系表逐页看、报告数字与 summary.csv 抽查 ≥5 个。
5. 把 report.pdf、outputs/figures/*.png、results/metrics、docs 拷进 repo/SuGaR → commit → push（远程 github，分支 dnc）。
6. 交付：report.pdf 路径 + https://github.com/Find-Yit/sugar-dnc-zju（分支 dnc）。

## 未完成清单
- P13 分析、终稿报告、图 fig1–fig7 重生成、README_ZH 更新、最终 commit/push、SELF_CHECK_阶段3f / 阶段5_终稿。
