# SELF_CHECK — 阶段 6：A100 提取端扩展实验合并进最终报告

> 执行者 10。2026-09-13 10:36 → 10:52（硬性截止 11:05，提前 13 分钟完成）。
> **只用 CPU 与 `tools/doc_env`；未使用 GPU；未 pip install；未删除任何文件；未 commit；
> 未改 `scripts/eval/` 与 `repo/` 下任何代码**（只往 `repo/SuGaR/` 新增了 A100 的只读产物副本）。

---

## 0. 交付物

| 项 | 值 |
|---|---|
| **最终 PDF** | `/scratch/e1351071/zju_test/outputs/reports/report.pdf`（6.1 MB） |
| **页数** | **34 页**（A4 纵向 595×842 点；合并前 26 页，新增 8 页） |
| **第 6 章页码** | **第 18–23 页**（6.1 硬件与可比性 p18｜6.2 来源声明 p19｜6.3 定量结果 p19–20｜6.4 可视化 p20–22｜6.5 结果解读 p22–23） |
| 源文件 | `outputs/reports/report.docx`（51.6 MB） |
| 图 / 表 | **12 图 16 表**（合并前 8 图 15 表） |
| 联系表 | `outputs/previews/report/全部页面联系表.png`（另有 `第01页.png`…`第34页.png`） |
| 自动检查记录 | `notes/doc_toolchain/SELF_CHECK_report.md` |
| 占位符清单 | `outputs/reports/report_占位符清单.json`（placeholders = 0，missing = 0） |

## 1. validate_artifacts.py 结果 —— **通过**

| 文件 | 类型 | 页数 | 中文占比 | 替代字符 | 自动检查 |
|---|---|---|---|---|---|
| `report.docx` | DOCX | — | 44.3% | 0 | **通过** |
| `report.pdf` | PDF | **34** | 46.1% | 0 | **通过** |

- 无文本页面：`[]`（无空白页）；替代字符 `�`：0（无中文方框）
- 中文占比较阶段 5（42.2%）**上升**（46.1%），因为新增第 6 章以中文叙述为主。

复算命令：

```bash
cd /scratch/e1351071/zju_test
tools/doc_env/bin/python .agents/skills/scientific-deliverables/scripts/validate_artifacts.py \
  outputs/reports/report.docx outputs/reports/report.pdf \
  --preview-root outputs/previews --report notes/doc_toolchain/SELF_CHECK_report.md \
  --font assets/fonts/NotoSansCJKsc-Regular.otf
```

## 2. 禁用措辞扫描 —— **0 命中**

```bash
tools/doc_env/bin/python - <<'PY'
import pymupdf
d = pymupdf.open("outputs/reports/report.pdf")
bad = ["尚未生成","分析未完成","待 Fable","待Fable","**","未能从","未能解析"]
hits = {}
for i,pg in enumerate(d,1):
    t = pg.get_text()
    for b in bad:
        if b in t: hits.setdefault(b,[]).append(i)
print(d.page_count, hits)      # -> 34 {}
PY
```

结果：`34 {}`。阶段 5 遗留的 P13「该部分见附录表格，分析未完成」已消失（P13 已写入分析文件并原文插入）。

## 3. 改动清单

### 3.1 数据复制（原样，未改一字）

| 源 | 目标 1 | 目标 2 |
|---|---|---|
| `repo/SuGaR_a100/results_a100/` | `outputs/a100/results_a100/` | `repo/SuGaR/results_a100/` |
| `repo/SuGaR_a100/patches_a100/` | `outputs/a100/patches_a100/` | `repo/SuGaR/patches_a100/` |
| `repo/SuGaR_a100/scripts_a100/` | `outputs/a100/scripts_a100/` | `repo/SuGaR/scripts_a100/` |
| `repo/SuGaR_a100/A100_DELIVERY.md` | `outputs/a100/A100_DELIVERY.md` | `repo/SuGaR/A100_DELIVERY.md` |

复算（全部 `OK`，103 个文件）：

```bash
for d in results_a100 patches_a100 scripts_a100; do
  diff -rq repo/SuGaR_a100/$d outputs/a100/$d && diff -rq repo/SuGaR_a100/$d repo/SuGaR/$d
done
find outputs/a100 -type f | wc -l          # -> 103
git -C repo/SuGaR status --porcelain       # -> 只有 4 行 "??"（新增未跟踪），无 M/D
```

### 3.2 `scripts/deliverables/make_report.py`（唯一被修改的脚本）

| # | 改动 | 说明 |
|---|---|---|
| 1 | 新增 `import csv` | 读 A100 的 summary.csv |
| 2 | 图表编号常量重排 | `FIG_A100_FRAG/NORMAL/SKY/SWEEP = 8,9,10,11`，`FIG_FAIL` 8→12，`N_FIGURES` 8→12；`TAB_A100 = 13`，`TAB_LEGACY/ARTIFACT/METRICDEF` 13/14/15→14/15/16，`N_TABLES` 15→16 |
| 3 | 新增 `load_a100_summary()` / `a100_rows()` / `_a100_cell()` / `a100_unified_para()` | 从 `outputs/a100/results_a100/metrics/summary.csv` 读 7 行 × 10 列并算相对基线百分比；`a100_unified_para()` 从 P14 原文抠出最后一段供摘要与结论复用 |
| 4 | 新增第 6 章 | `6 扩展实验：提取端改进的完整扫描（4×A100）`，含 6.1–6.5 五小节 |
| 5 | 后续章节重编号 | `6 失败案例与代价`→`7`；`6.1 方法本身的局限`→`7.1`；`7 结论与下一步`→`8` |
| 6 | 封面「改动点」加第 ③ 条 | 「③ 提取端扩展（A100）：q=0 清洗 + M2-B DBSCAN 剪枝 + M1+a 背景 Poisson 深度，碎片 −43%，渲染指标不变」 |
| 7 | 摘要末尾 + 第 8 章结论 | 各插入 P14 最后一段「与训练端结果的统一解读」**原文** |
| 8 | 附录 A 新增 `A.5` | 指向 `results_a100/notes/COMMANDS.md`、`scripts_a100/README.md`、`patches_a100/PARAMS.md`、`COVERAGE_METRIC.md`、`NEGATIVE_RESULTS.md` |
| 9 | 附录 C 路径表加一条 | 指向 `outputs/a100/` 全部产物 |

> 说明：第 6 章的图**直接从 `outputs/a100/results_a100/figures/` 读**，不经 `figures_manifest.json`，
> 因此**没有重跑 `make_figures.py`**（按指令要求）。

### 3.3 `repo_release/README_ZH.md`

在 §1「结果摘要」的提取端扫参表之后新增 `### 提取端扩展实验（A100）` 小节：一行结果摘要、
5 行结果表、来源声明、负结果、产物与复现路径表（指向 `results_a100` / `patches_a100` / `scripts_a100`）。

## 4. 第 6 章内容对照（逐条核对任务要求）

| 要求 | 落地位置 | 核对 |
|---|---|---|
| (a) P14 按 `## P14` 原文插入，不改写 | 6.5 结果解读（p22–23），走已有 `analysis()` 机制（`load_analysis()` 的 `^##\s+(P\d+)` 正则已支持） | **PASS**，4 段全部逐字 |
| (b) 表：7 行 × 指定 10 列 + 相对 base_d10_q01 百分比 | **表 13**（p19–20） | **PASS** |
| (c) 4 张图，中文图注 + 「由 A100 机器生成，数据见 results_a100」 | **图 8**（碎片着色对比，p20）、**图 9**（法向 6×4，p21）、**图 10**（天空封口证据，p21）、**图 11**（扫参，p22） | **PASS**，4 张均以 `（来源：由 A100 机器生成，数据见 results_a100）` 结尾 |
| (d) 来源声明（M1 移植自 Frosting / M2 来自 2d-sugar / 自有 M1+a、M2-B、q 参数化） | 6.2（p19），5 条 bullet + 补丁路径 | **PASS** |
| (e) 硬件与可比性（复用 hopper `coarse_base_seed0` .pt，未重训） | 6.1（p18），含 2094 vs 2137 的 2% 差异说明与「不跨机器相减」的基准约定 | **PASS** |

### 表 13 的数值（逐个与 `summary.csv` 核对，均一致）

| 配置 | D_fg/D_bg | q | 剪枝 | 连通分量 | 碎片 | 最大分量占比 | 边界边 | G1 中位(abs) | 面数 |
|---|---|---|---|---|---|---|---|---|---|
| 原版基线 | 10/10 | 0.1 | — | 2,184 | 2,094 | 42.3% | 38,450 | 0.00405 | 370,284 |
| 仅 q=0 | 10/10 | 0.0 | — | 1,651 (−24.4%) | 1,628 (−22.3%) | 93.0% (+119.8%) | 11,339 (−70.5%) | 0.00418 (+3.2%) | 331,769 (−10.4%) |
| Frosting 原样自动深度 + q=0 | 10/10 | 0.0 | — | 1,645 (−24.7%) | 1,622 (−22.5%) | 93.0% | 11,353 (−70.5%) | 0.00417 (+3.0%) | 331,767 |
| M1+a 背景 D=9 + q=0 | 10/9 | 0.0 | — | 1,478 (−32.3%) | 1,456 (−30.5%) | 92.9% | 11,033 (−71.3%) | 0.00417 (+3.0%) | 321,501 |
| M2 原版「只留最大簇」+ q=0（失败案例） | 10/10 | 0.0 | 59.7% | 1,302 (−40.4%) | 1,275 (−39.1%) | 92.9% | 12,375 (−67.8%) | 0.00417 (+3.1%) | 307,721 |
| M2-B p95 + q=0 | 10/10 | 0.0 | 14.2% | 1,434 (−34.3%) | 1,412 (−32.6%) | 94.0% | 11,352 (−70.5%) | 0.00407 (+0.5%) | 325,562 |
| **M2-B + M1+a（最优）** | 10/9 | 0.0 | 14.2% | **1,216 (−44.3%)** | **1,194 (−43.0%)** | **93.9%** | **11,225 (−70.8%)** | **0.00407 (+0.5%)** | 313,971 |

核心数字复算：`1194 / 2094 − 1 = −42.98%` ≈ **−43%**（与 P14 与 A100_DELIVERY.md 一致）。

## 5. 逐页视觉验收后修掉的排版问题

| # | 问题 | 处理 | 复核 |
|---|---|---|---|
| G1 | 6.2 的 5 条 bullet 里 `**M1（自动 Poisson 深度）**` 的星号被当字面量打印（`bullets()` 不解析 markdown） | 改为 `for _it in [...]: rich_para(doc, _it, style="List Bullet")` | p19 已正常加粗 |
| G2 | 6.1 末句 `而**被改动的对象完全相同**` 同样漏出星号 | `para()` → `rich_para()` | p18 已正常 |
| G3 | 表 13「最大分量占比」列过窄，`（+119.8%）` 的右括号单独折到第三行 | 先试两轮调列宽（1.5→1.8→2.15 cm）**完全无效** —— `add_table()` 的 `col_widths` 在本表未生效（`table.autofit` 未关，LibreOffice 按内容重算）。改为**不动列宽、只缩内容**：`_a100_cell()` 里的全角括号 `（）` 换半角 `()`，每个全角括号省约 0.24 cm | **已修复**，p19–20 全部单元格整齐折成两行，无孤立括号 |

复核方式：PDF 全文扫 `**` 由 1 页命中降为 **0 页命中**；并逐页看了 `第18–22页.png` 原图。
G1、G2、G3 全部已修复并逐页复核（`第18–22页.png` 原图）。注意 `add_table()` 的 `col_widths` 对本表不生效，后续若要改该表版式，应改内容长度或先关掉 `table.autofit`。

## 6. 剩余问题（已复核，判定为可接受）

1. **图 8（碎片着色对比）分辨率偏低**：源 PNG 由 A100 机器生成，2×4 小图放进 16.2 cm 宽后，
   红色碎片斑点较小，投影时需放大看；未重绘（指令要求不重跑绘图、且源图只读）。
2. **表 13 第 5 行是失败案例但数字「看起来好」**：M2 原版「只留最大簇」剪掉 59.7% 高斯后碎片降到 1275，
   但它删了 80.7% 的背景，属失败案例。表里靠配置名标注「（失败案例）」，成因在 6.5（P14 原文）第 3 条说明；
   表注未重复该说明。
3. **DOCX 体积 22.0 MB → 51.6 MB**：新增 4 张 A100 大图所致；PDF 只有 6.1 MB，交付无影响。
4. **第 6 章未给渲染指标列**：按构造 PSNR/SSIM/LPIPS 与 λ=0 组完全相同，表注已说明；
   网格渲染 PSNR（正确的提取端渲染指标）A100 侧未做（B13 未提供），P14 第 3 段已如实写明。
5. **跨机器可比性**：hopper 基线碎片 2137、A100 基线 2094，相差 2%。报告一律用 A100 自己的基线做分母，
   不跨机器相减；这一点在 6.1 与表注两处都写了。
7. **表 13 的百分比用半角括号**，与报告其余表格的全角括号风格略有不一致（为修 G3 所必需）；
   另外「最大分量占比」列的相对变化（如 `+119.8%`）是**在百分数上再算相对变化**，
   严格说用百分点（`+50.7pp`）更准确，但任务明确要求「相对 base_d10_q01 的百分比」，故按要求保留。
8. **PDF 中仍有 5 页出现「未完成」字样**（第 1、8、12、25、32 页），经逐处核对**全部是合理用法**，
   与第 6 章无关：p1 是封面在解释「未跑出结果的位置显示未完成」这条约定；p8/p12/p32 是
   `3DGS 7k(参考)` 参考行本来就没有训练/几何指标；p25 是「哪一处差异是主因需要逐项消融，本次未完成」
   这句如实陈述。任务点名的三个措辞「尚未生成 / 分析未完成 / 待 Fable」命中数为 **0**。
6. `outputs/reports/` 下的历史文件（`report_final.*`、`report_layout_check.*`、`report_布局验证_baseline.*`）
   按「不删文件」规则全部保留；**注意 `report_final.*` 现在已比 `report.*` 旧（不含第 6 章）**，
   若要对外交付请用 `report.pdf`。

## 7. 审计

```bash
cd /scratch/e1351071/zju_test
git -C repo/SuGaR status --porcelain          # 仅 4 行 "??"（A100 产物新增），分支 dnc，无 M/D
ls -l --time-style=+%H:%M scripts/eval/*.py   # 全部 ≤ 09:28，本阶段未触碰
find outputs/a100 -type f | wc -l             # 103
```

- 本阶段新增：`outputs/a100/**`（103 文件）、`repo/SuGaR/{results_a100,patches_a100,scripts_a100,A100_DELIVERY.md}`、
  本文件。
- 本阶段修改：`scripts/deliverables/make_report.py`、`repo_release/README_ZH.md`、
  `outputs/reports/report.{docx,pdf}`、`outputs/reports/report_占位符清单.json`、
  `outputs/previews/report/**`、`notes/doc_toolchain/SELF_CHECK_report.md`。
- **未删除任何文件、未 commit、未 push、未使用 GPU、未 pip install、未改 `scripts/eval/` 与 `repo/` 下的代码。**
