# SELF_CHECK — 阶段 5 终稿（report.pdf 定稿）

> 执行者 5（opus-executor）。终稿阶段 2026-09-13 09:32 → 10:25。
> **只用 CPU 与 `tools/doc_env`，未使用 GPU；未 pip install；未删除任何文件；未 commit；未改 `scripts/eval/` 与 `repo/` 下任何代码。**

---

## 0. 交付物

| 项 | 值 |
|---|---|
| **最终 PDF** | `/scratch/e1351071/zju_test/outputs/reports/report.pdf` |
| 页数 | **26 页**（A4 纵向，595×842 点） |
| 源文件 | `outputs/reports/report.docx`（22.0 MB），另存同内容副本 `report_final.docx/.pdf` |
| 图 / 表 | **8 图 15 表**（编号固定，正文交叉引用不会错位） |
| **联系表** | `/scratch/e1351071/zju_test/outputs/previews/report/全部页面联系表.png`（另有 `第01页.png`…`第26页.png`） |
| 自动检查记录 | `notes/doc_toolchain/SELF_CHECK_report.md` |
| 占位符清单 | `outputs/reports/report_占位符清单.json` |

## 1. validate_artifacts.py 结果

| 文件 | 类型 | 页数 | 中文占比 | 替代字符 | 自动检查 |
|---|---|---|---|---|---|
| `report.docx` | DOCX | — | 40.4% | 0 | **通过** |
| `report.pdf` | PDF | 26 | 42.2% | 0 | **通过** |

- 无文本页面：`[]`（没有空白页）
- 替代字符（`�`）：0，即没有中文方框 / 乱码
- 中文占比 40%+ 是因为附录 A 嵌入了 4 大段逐字命令、附录 C 全是路径、附录 E 是英文原始题名——
  这些正是 `SKILL.md` 明确允许的例外（URL、软件名、公式变量、参考文献原题）。正文叙述部分为全中文。
  该脚本对占比无硬阈值，判定依据是"替代字符 / 越界对象 / 空白页"三项，全部为 0。

复算命令：

```bash
cd /scratch/e1351071/zju_test
tools/doc_env/bin/python .agents/skills/scientific-deliverables/scripts/validate_artifacts.py \
  outputs/reports/report.docx outputs/reports/report.pdf \
  --preview-root outputs/previews --report notes/doc_toolchain/SELF_CHECK_report.md \
  --font assets/fonts/NotoSansCJKsc-Regular.otf
```

## 2. 占位符是否全部消除

| 项 | 结果 |
|---|---|
| `【待 Fable 填写：…】` 橙色占位符 | **0 处**（P1–P12 全部由 `notes/REPORT_ANALYSIS_fable.md` 原文插入，未改写、未润色） |
| 尚未生成的数据 / 图 | **0 处** |
| 唯一未完成项 | **P13**（Frosting 自动 Poisson 深度的结果解读）截至 10:25 尚未写入分析文件；正文按协调者指定的措辞输出红色一行：**“该部分见附录表格，分析未完成。”**，该节的表 12 与图 7 的数字均已生成 |

P13 到位后只需重跑两条命令即可自动替换（脚本会检测 `## P13` 小节）：

```bash
tools/doc_env/bin/python scripts/deliverables/make_report.py --out outputs/reports/report.docx \
  --labels "基线 λ=0" "本工作 λ=0.05" "本工作 λ=0.2" "λ=0.2+detach" "官方 DNC"
LC_ALL=C.UTF-8 scripts/deliverables/libreoffice.sh --headless --convert-to pdf \
  --outdir outputs/reports outputs/reports/report.docx
```

## 3. 报告结构（按协调者要求调整后）

```
封面（含数据来源声明框）
摘要                      ← P1 原文
预注册的可检验假设 H1/H2/H3 ← P2 原文
1 环境与数据               表 1 环境 / 表 2 数据划分
2 复现流程与命令
3 改动动机与实现            （开头说明本工作有训练端 + 提取端两处改动）
  3.1 动机 / 3.2 思路来源与自己的贡献 / 3.3 损失定义（图 1 公式）/ 3.4 代码改动范围（表 3）
4 实验设置与对照            表 4 配置差异（含留档组）
  4.1 过程记录：首轮 λ 组被中断与重跑   表 5 残留记录 + P4 原文 + 官方对照与提取端扫参的时间线
5 结果
  5.1 定量结果             表 6 渲染 / 表 7 几何精度 / 表 8 拓扑 / 表 9 平滑度 / 表 10 训练代价 + P5 原文
  5.2 定性结果             图 2 指标汇总 / 图 3 渲染与误差 / 图 4 网格法向 / 图 5 训练曲线 / 图 6 DNC 中间量 + P6 原文
  5.3 官方 dn_consistency 对照   表 11 实现差异清单（四条）+ P12 原文
  5.4 提取端改动：Frosting 的自动 Poisson 深度  表 12 扫参 + 图 7 扫参柱状图 + P13（待补）
6 失败案例与代价            图 8 失败案例特写 + P7 / P8 / P9 原文 + 6.1 方法本身的局限
7 结论与下一步              P10 / P11 原文
附录 A 完整命令（env.sh / setup 脚本 / 三步命令 / 评测命令，全部原样提取）
附录 B 留档对照：未打补丁的原版基线   表 13
附录 C 日志与产物路径      表 14
附录 D 评测指标的精确定义   表 15
附录 E 参考文献（SuGaR / 3DGS / 2DGS / T&T / Poisson / LPIPS）
```

**训练端主表改为五列**：基线 λ=0 / 本工作 λ=0.05 / 本工作 λ=0.2 / λ=0.2+detach / 官方 DNC
（3DGS 7k 作参考行）；阶段 2 的未打补丁 `coarse_baseline` 已移到附录 B。

## 4. 图（全部重生成，中文标注、标方向与单位）

| 图号 | 文件 | 内容 |
|---|---|---|
| 图 1 | `fig_公式_DNC.png` | L_DNC 定义与掩码（mathtext 渲染） |
| 图 2 | `fig1_指标对比.png` | 7 个小图 × **6 组**（五组训练侧 + λ=0+Frosting 提取），含 Δ 标注与方向箭头 |
| 图 3 | `fig2_渲染对比.png` | 4 视角 ×（GT + 5 组渲染 + 5 组误差），误差图共用 turbo 色条（上限 0.25） |
| 图 4 | `fig3_网格法向对比.png` | 4 视角 × **6 组**网格法向 |
| 图 5 | `fig4_训练曲线.png` | λ=0.05 / λ=0.2 / detach 的 L_dnc、SDF 法向损失、有效像素占比、总损失 |
| 图 6 | `fig5b_DNC中间量_各λ对比.png` | 三行（λ=0.05 / λ=0.2 / detach）× D / N / N_d / 掩码，**第 15000 迭代** |
| 图 7 | `fig7_提取端扫参.png` | **新增**：碎片数、G1 中位数、G5 二面角、最大分量占比随 D / quantile 变化 |
| 图 8 | `fig6_失败案例特写.png` | 同一地面区域在 λ=0 / 0.05 / 0.2 下的渲染、误差与网格法向（λ=0.05 地面灰板、λ=0.2 坍缩） |

`figures_manifest.json` 里 8 条记录全部 `ok`。

## 5. 我逐页看联系表后发现并修掉的排版问题

| # | 问题 | 处理 | 复核 |
|---|---|---|---|
| F1 | 五组对照后表格列名被挤断（“λ=0.2 +/detach”“官方 dn_consisten/cy”跨行拆词） | 报告里改用短标签（`--labels "基线 λ=0" … "官方 DNC"`），并重排表 6–表 10 的列宽 | 第 10、11 页已正常 |
| F2 | 表 11 的表题落在页底、表格被推到下一页 | 表题段落加 `keep_with_next` | 第 14/15 页已同页 |
| F3 | 表 12（提取端扫参）列宽合计 18.0 cm > 版心 16.2 cm，单元格被严重挤压 | 列宽改为合计 16.2 cm，字号 7.4→7.0 | 第 16 页可读 |
| F4 | 图编号常量与实际插入顺序不符（5.4 的扫参图排在第 6 章的失败案例图之前） | 交换 `FIG_SWEEP`/`FIG_FAIL` 常量，使“见图 N”与实际编号一致 | 图 7=扫参、图 8=失败案例 |
| F5 | 表 10 训练代价里官方组的 λ 显示“未完成” | 官方训练器把权重写成 `dn_consistency_factor`，补上该字段的读取 | 现显示 0.05 |
| F6 | 封面“对照组”只写了三组、“改动点”只写了训练端 | 封面改为五组 + 提取端扫参，改动点写两条 | 第 1 页已更新 |
| （前一轮已修并保持有效） | 表格跨页把一行切开、长脚注撑宽画布、`**` 被当字面量、参考行标签重复等 9 项 | 见 `notes/SELF_CHECK_阶段5_骨架.md` §5 | — |

## 6. 剩余问题（已复核，判定为可接受或需他人补齐）

1. **P13 未写入**（唯一硬缺口）。正文按约定写“该部分见附录表格，分析未完成”，表 12 与图 7 的数字完整。
2. 表 12 的“（±x%）”变化量在窄列里会折到第二行，个别行的右括号单独成行。数值本身完整可读；
   继续压缩字号会低于 7 pt，不利投影，故保留。
3. 大图前偶有半页留白（Word 整图不可分页的固有行为），例如第 12、17 页。
4. 表 6 的“3DGS 7k(参考)”与表 14 的长路径仍会折行，不影响可读性。
5. `outputs/reports/` 下留有历史文件（`report_布局验证_baseline.*`、`report_layout_check.*`、
   `report_final.*`）。按“不删文件”规则全部保留；`report_final.*` 与 `report.*` 内容相同。
6. 图 2 中 `base_seed0_pd8`（D=8）未进入六组对比图，只在表 12 与图 7 里；因为它是提取端扫参的一员，
   放进训练侧对比图会造成误读。

## 7. 审计

```bash
cd /scratch/e1351071/zju_test
git -C repo/SuGaR status --porcelain      # 输出为空（主会话 09:19 已 commit 中期快照）
ls -l --time-style=+%H:%M scripts/eval/*.py
```

说明：`scripts/eval/summarize.py`（mtime 09:11）与新增的 `scripts/eval/blocktime_from_logs.py`
（09:28）是**别的执行者**在阶段 3e/3f 里改/加的（见 `notes/SELF_CHECK_阶段3f_poisson.md` 的
“summarize.py 的最小改动”一节）；本执行者终稿阶段对 `scripts/eval/` 与 `repo/` 全程**只读**。

- 本阶段新增 / 更新：`scripts/deliverables/{_deliv_common.py,make_figures.py,make_report.py}`、
  `outputs/figures/fig{1..7}_*.{png,pdf}`、`outputs/reports/report.{docx,pdf}`（及 `report_final.*`）、
  `outputs/previews/report/**`、`notes/doc_toolchain/SELF_CHECK_report*.md`、
  `repo_release/README_ZH.md`、本文件。
- **未删除任何文件、未 commit、未 push、未使用 GPU、未改 `scripts/eval/` 与 `repo/` 下的代码。**
