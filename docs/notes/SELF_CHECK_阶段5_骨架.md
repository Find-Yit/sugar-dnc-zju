# SELF_CHECK — 阶段 5 骨架（图表脚本 / 报告脚本 / GitHub 脚手架）

> 执行者 5（opus-executor）。起 2026-09-13 07:05，止 08:12（中间 06:49 PBS 作业被强杀、会话中断，
> 07:59 在新作业 hopper-13 上恢复，磁盘产物全部完好，未重做已完成部分）。
> **全程只用 CPU 与 `tools/doc_env`，没有占用任何 GPU；没有 pip install 任何包；没有删除任何文件；没有 git commit。**

---

## 0. 验收条目总表

| # | 验收条目 | 实际结果 | 判定 |
|---|---|---|---|
| A1 | `scripts/deliverables/make_figures.py` 存在且可参数化 run 列表 | 791 行，`--runs/--labels/--only/--outdir/--dpi/--max_views/--dnc_vis_run/--crop/--manifest` | PASS |
| A2 | fig1 指标对比（3 渲染 + 4 几何小图，分组柱状，数值标注，方向与单位） | `outputs/figures/fig1_指标对比.png`（4522×2496，300 dpi）+ .pdf | PASS |
| A3 | fig2 渲染对比拼图（4 视角 ×（GT + 各 run 渲染 + 各 run 误差），误差共用色条） | `fig2_渲染对比.png`，右侧共用 turbo 色条，上限 0.25（取自 render_*.json 的 `config.error_heatmap_vmax`） | PASS |
| A4 | fig3 mesh 法向拼图（4 视角 × 各 run） | `fig3_网格法向对比.png` | PASS |
| A5 | fig4 训练曲线（L_dnc 与 sdf_better_normal_loss；无 λ 组时优雅降级） | 4 联图 (a)L_dnc (b)SDF 法向损失 (c)有效像素占比 (d)总损失；λ=0 组无 dnc_log 时该组在 (a)(b)(c) 缺席并在脚注写明 | PASS |
| A6 | fig5 DNC 中间量（λ 最大组最后一个迭代的 D/N/N_d/mask 四联图） | `fig5_DNC中间量.png`；另加 fig5b 各 λ 对比（超出要求） | PASS |
| A7 | fig6 失败案例特写，`--crop "run,view,x0,y0,x1,y1"` 接口 | 支持重复 `--crop`，`run` 可写 `ALL`；不给时用 3 个内置默认裁剪框 | PASS |
| A8 | 无数据时优雅跳过并提示 | fig5/fig5b/fig2/fig3/fig6 在缺数据时均输出 `[SKIP] <原因>` 并写进 manifest | PASS |
| A9 | `scripts/deliverables/make_report.py` 生成 `outputs/reports/report.docx` | 1087 行；8 图 12 表 20 页 | PASS |
| A10 | 数字一律从 CSV/JSON 读入，不手写 | 全部经 `_deliv_common.load_summary/load_json`；脚本内无任何实验数值常量（见下 §4 的自查命令） | PASS |
| A11 | 分析性文字写成 `【待 Fable 填写：…】` | 11 处，见 §3；另有 1 处红色`【尚未生成：…】` | PASS |
| A12 | 报告注明 4 项约定（数据来源 / CUDA 非确定 / decimation 语义 / 掩码保护 / 阶段 2 留档） | 5 项全部写入，见 §2 | PASS |
| A13 | 转 PDF | `LC_ALL=C.UTF-8 scripts/deliverables/libreoffice.sh` → `outputs/reports/report.pdf`（20 页） | PASS |
| A14 | `validate_artifacts.py` 验收 | docx 33.2% 中文 / pdf 35.3% 中文，替代字符 0，无空白页，自动检查「通过」 | PASS |
| A15 | 人眼看过联系表 | 两份报告的联系表逐页看过，问题清单见 §5 | PASS |
| A16 | GitHub 脚手架在 `repo_release/`（不是 repo/SuGaR） | README_ZH.md / .gitignore / scripts/reproduce_all.sh | PASS |
| A17 | 新增"过程记录"章节（首轮 λ 组 14900/15000 被杀 + 单卡顺序重跑） | 报告 §4.1 + 表 5（数字由脚本从残留目录实测） | PASS |
| A18 | 不改 repo/SuGaR、repo/SuGaR_dev、scripts/eval | 我对 `repo/` 只有读操作；`scripts/eval/` mtime 全部停在 06:14 未变；`repo/SuGaR` 现有的 3 处改动是别的执行者 07:59 打的 DNC 补丁（见 §6） | PASS |

---

## 1. 脚本清单与运行命令

| 文件 | 行数 | 作用 |
|---|---|---|
| `scripts/deliverables/_deliv_common.py` | 378 | 共用：路径常量、中文字体注册、配色、run 名归一化（与 `scripts/eval/summarize.py` 的 `normalize()` 一致）、`RunData` 数据类（render/geometry/meshvis JSON + summary 行 + dnc_log + dnc_vis + 训练日志） |
| `scripts/deliverables/make_figures.py` | 791 | fig1–fig6（+fig5b）+ `figures_manifest.json` |
| `scripts/deliverables/make_report.py` | 1087 | `report.docx` + `report_占位符清单.json` + 公式图 |

复算命令（全部用文档 venv，不要 activate）：

```bash
cd /scratch/e1351071/zju_test

# 1) 图（默认三组；三组评测跑完后重跑这一条即可）
tools/doc_env/bin/python scripts/deliverables/make_figures.py

# 2) 报告
tools/doc_env/bin/python scripts/deliverables/make_report.py

# 3) 转 PDF —— 注意 LC_ALL（见 §5 的 [DEVIATION-1]）
LC_ALL=C.UTF-8 scripts/deliverables/libreoffice.sh --headless --convert-to pdf \
  --outdir outputs/reports outputs/reports/report.docx

# 4) 逐页渲染验收
tools/doc_env/bin/python .agents/skills/scientific-deliverables/scripts/validate_artifacts.py \
  outputs/reports/report.docx outputs/reports/report.pdf \
  --preview-root outputs/previews --report notes/doc_toolchain/SELF_CHECK_report.md \
  --font assets/fonts/NotoSansCJKsc-Regular.otf
```

本次另外跑过的两组（用于排版验收与失败记录）：

```bash
# 排版验收版：用 coarse_baseline 把 8 个图槽位全部填满，专门检查版式
tools/doc_env/bin/python scripts/deliverables/make_figures.py \
  --runs coarse_baseline --labels "基线(阶段2 留档)" --outdir outputs/figures/_布局验证_baseline
tools/doc_env/bin/python scripts/deliverables/make_figures.py \
  --only fig5 fig5b --outdir outputs/figures/_布局验证_baseline          # 清单是增量合并的
tools/doc_env/bin/python scripts/deliverables/make_report.py \
  --runs coarse_baseline --labels "基线(阶段2 留档)" --ref_runs vanilla3dgs7k \
  --manifest outputs/figures/_布局验证_baseline/figures_manifest.json \
  --out outputs/reports/report_layout_check.docx

# 首轮被中断的 λ 组（失败记录素材）
tools/doc_env/bin/python scripts/deliverables/make_figures.py \
  --runs coarse_base_seed0 coarse_dnc005_killed_0649 coarse_dnc02_killed_0649 \
  --labels "基线(λ=0)" "首轮 λ=0.05（被中断）" "首轮 λ=0.2（被中断）" \
  --outdir outputs/figures/_首轮被中断记录
```

---

## 2. 产物清单

### 2.1 图（`outputs/figures/`，300 dpi PNG + 矢量 PDF）

| 文件 | 字节(PNG) | 本次数据来源 |
|---|---|---|
| `fig1_指标对比.png/.pdf` | 485,969 | summary.csv（目前只有 `base_seed0` 一组有数，λ 两组显示"无数据"） |
| `fig2_渲染对比.png/.pdf` | 7,049,604 | `outputs/vis/coarse_base_seed0/`（λ 两组显示"缺图"） |
| `fig3_网格法向对比.png/.pdf` | 2,878,146 | 同上 |
| `fig4_训练曲线.png/.pdf` | 802,789 | `outputs/runs/coarse_dnc02/dnc_log.csv` + 三组训练日志 |
| `fig5_DNC中间量.png/.pdf` | 1,236,806 | `outputs/runs/coarse_dnc02/dnc_vis/`（重跑中的 λ=0.2） |
| `fig5b_DNC中间量_各λ对比.png/.pdf` | 2,620,825 | **陈旧文件**：manifest 里状态是 `skipped`（当前只有 1 组有 dnc_vis），报告不会引用它；λ=0.05 重跑完后重跑 make_figures 会覆盖 |
| `fig6_失败案例特写.png/.pdf` | 3,301,509 | `outputs/vis/coarse_base_seed0/` 的 3 个默认裁剪框 |
| `fig_公式_DNC.png/.pdf` | 321,642 | matplotlib mathtext 渲染的 L_DNC 定义（由 make_report.py 生成） |
| `figures_manifest.json` | — | 7 条记录：fig1/2/3/4/5/6 = ok，fig5b = skipped |

子目录：`outputs/figures/_布局验证_baseline/`（7 图，全 ok）、`outputs/figures/_首轮被中断记录/`（6 图）。

### 2.2 报告

| 文件 | 说明 |
|---|---|
| `outputs/reports/report.docx` / `.pdf` | **正式骨架**，默认三组；20 页 / 8 图 / 12 表 |
| `outputs/reports/report_占位符清单.json` | 11 个待填写 + 1 个尚未生成 + 图表标题顺序 |
| `outputs/reports/report_layout_check.docx` / `.pdf` | 排版验收副本：8 个图槽位全部有图，用来检查版式 |
| `outputs/reports/report_布局验证_baseline.*` | 同上的**旧文件名版本**（文件名含中文导致 LibreOffice 报错，见 §5）。按规则未删除，可忽略；如需清理请主会话授权 |

### 2.3 联系表（逐页预览）

- **正式骨架**：`outputs/previews/report/全部页面联系表.png`（20 页，单页图 `第01页.png`…`第20页.png`）
- **排版验收**：`outputs/previews/report_layout_check/全部页面联系表.png`（20 页）
- 自动检查记录：`notes/doc_toolchain/SELF_CHECK_report.md`、`notes/doc_toolchain/SELF_CHECK_report_layout_check.md`

### 2.4 GitHub 脚手架（`repo_release/`，11.5 KB + 1 KB + 6.7 KB）

| 文件 | 内容 |
|---|---|
| `repo_release/README_ZH.md` | 改动说明与 L_DNC 公式、预注册假设、环境安装（torch 2.4.1+cu121 / pytorch3d wheel / 两个 CUDA 扩展编译 / PIP_CONSTRAINT 护栏）、数据获取、五步逐条命令、指标定义表、可复现性说明、引用（SuGaR / 3DGS / 2DGS / T&T / Poisson） |
| `repo_release/.gitignore` | 排除 `output/ outputs/ *.pt *.ply *.pth data/ ckpts/ wheels/ __pycache__ logs/`，并显式放行 `results/ figures/ report.pdf` |
| `repo_release/scripts/reproduce_all.sh` | 串起 gs→coarse→mesh→eval→report；路径全用变量（`PROJ_ROOT/SUGAR_DIR/SCENE/GS_ITER/...`）；支持 `GPUS=0,1` 并行、`STAGES=` 选段、`DRY_RUN=1` 空跑、已存在产物自动跳过 |

`reproduce_all.sh` 自检：`bash -n` 通过；`DRY_RUN=1` 空跑打印出的命令与 `notes/RUNLOG_s34.md` 里实际执行的命令逐字段一致（场景、`-c` 结尾斜杠、`-i 7000`、`-l 0.3 -d 200000`、`--seed 0`、`--dnc_start 9000`）。

---

## 3. 待 Fable 填写的占位符（11 处，橙色加粗，PDF 里一眼可见）

1. 摘要的结论句（必须与表 6–表 10 的数字一致）
2. H1 / H2 / H3 逐条结论（H1 看表 7、表 8；H2 看表 6；H3 看图 5 的 (a)(b)）
3. `coarse_baseline`（未打补丁）与 λ=0 组的差异是否只在 CUDA 非确定性量级内
4. 过程记录补充：首轮中断的具体时刻、双卡→单卡、重跑起止时间及其对墙钟对比的影响
5. 定量结果解读（表 6–表 10）
6. 定性结果解读（图 3 渲染/误差 + 图 4 网格法向，与表 8 碎片数、表 9 二面角互证）
7. 失败案例分析（对照图 8）
8. 代价分析（表 6 与表 10）
9. λ 敏感性（若 λ=0.2 退化，必须写明并给机理）
10. 结论 3–5 条，每条可追溯到表/图编号
11. 下一步

另有 1 处红色`【尚未生成：图 7 …】`——`fig5b` 需要至少两组有 `dnc_vis`，等 λ=0.05 重跑完后重跑 make_figures + make_report 即自动消失。

---

## 4. 诚信条款的自查

- 报告封面有「数据来源声明」框：所有数值由脚本从 `outputs/metrics/summary.csv` 及
  `render_*.json / geometry_*.json / train_stats.json / dnc_log.csv` 自动读取，未经人工转录；
  读不到显示「未完成」。
- 报告正文明确写入的 5 条约定：
  1. 数字全部来自 summary.csv 与各 metrics JSON（封面声明 + 各表脚注）；
  2. CUDA 光栅化原子累加非确定，固定 seed 仍有微小差异，0.01 dB 量级不算显著（封面 + §2 + §6.1）；
  3. `-d 200000` 是**前景/背景各 20 万三角形**的目标，合并后约 37 万面（§2 红字勘误 + 表 8 上下文）；
  4. DNC 掩码含 `‖N_raw‖>0.1` 的低不透明度保护（§3.2 P3 + 公式图第 ③ 行 + 图 5(c) 注记）；
  5. 阶段 2 未打补丁的 `coarse_baseline` 仅作留档、不参与三组对照（§4 留档说明 + 各表「（参考）」标记）。
- 脚本无硬编码实验数值自查：

```bash
grep -nE '2[0-9]\.[0-9]{2,}|0\.8[0-9]{3}|[0-9]{3,},[0-9]{3}' \
  scripts/deliverables/make_figures.py scripts/deliverables/make_report.py
# 实测只命中 1 行：make_figures.py 第 12 行 docstring 里的 --crop 用法示例（裁剪框像素坐标）。
# 没有任何实验数值被写死。
```

---

## 5. 人眼看过联系表后的排版问题清单

已修（改完重新渲染确认）：

| # | 问题 | 处理 |
|---|---|---|
| L1 | 表格跨页时把一行从中间切开，下一页顶部出现没有表头的半行（原第 4/9/18/19 页） | `add_table()` 给表头行加 `w:tblHeader`（跨页重复表头）、每行加 `w:cantSplit`（行内不分页）；重渲染后第 3/4/9 页的续表都带表头 |
| L2 | fig1 的副标题与上排子图标题重叠 | suptitle 上移到 y=0.985、副标题 y=0.938、GridSpec top=0.845 |
| L3 | 单 run 时柱子占满整个子图，比例失衡 | `ax.set_xlim(-0.65, n-0.35)` |
| L4 | 长脚注把 `bbox_inches='tight'` 的画布横向撑开，图片两侧出现大片空白（fig3 曾被撑到 14.35 英寸） | `_footnote()` 按图宽自动折行（textwrap）+ `va="top"` |
| L5 | 拼图瓦片留白过多 | 瓦片高宽比改成与原图一致（546/979） |
| L6 | 曲线图无数据时仍残留坐标轴标签与注记 | `_blank_ax()` 清空 xlabel/ylabel，注记移进 `if drew` 分支 |
| L7 | 正文里的 `**加粗**` 被原样显示（python-docx 不解析 Markdown） | 去掉这两处 `**` |
| L8 | 参考行标签出现「3DGS 7k(参考)（参考）」重复 | `row_label()` 判断标签里已含「参考」就不再追加 |
| L9 | 「首轮被中断」表里混进了空的 `mesh_*_killed_*` 目录 | 跳过空目录 |

仍存在、判定为**可接受**（已复核，不打算再改）：

| # | 现象 | 判断 |
|---|---|---|
| L10 | 大图前会出现半页空白（Word 的整图不可分页所致，例如第 9→10 页） | 属于 Word 排版固有行为；缩小图反而损失可读性 |
| L11 | 续表偶尔只带 1 行过到下一页（表 2 的「几何参考」行） | 已有重复表头，不影响阅读 |
| L12 | 中文占比 33%–35%（远低于工具链样例的 96%） | 附录 A 里嵌入了 4 大段**逐字命令**、附录 B 全是路径、参考文献是英文原题名——这些正是 SKILL.md 允许的例外（URL、软件名、公式变量、参考文献原题）。去掉附录 A/B 的代码块后正文本体仍是全中文。`validate_artifacts.py` 对占比无硬阈值，自动检查判「通过」，替代字符 0、无空白页 |
| L13 | fig1 目前大面积显示「无数据」 | 三组评测未完成时的**正确**表现；λ 两组评测完成后重跑即消失 |

---

## 6. 未改动 / 未删除 的审计

```bash
cd /scratch/e1351071/zju_test
git -C repo/SuGaR status --porcelain
#  M sugar_trainers/coarse_density.py
#  M train_coarse_density.py
#  ?? sugar_utils/dnc_utils.py        <- 三者 mtime 均为 09-13 07:59
ls -l --time-style=+%H:%M scripts/eval/*.py scripts/eval/*.sh   # mtime 全在 05:54–06:14，未被我改动
```

**说明**：`repo/SuGaR` 工作区现在带着 DNC 补丁（上面 3 个文件），但那**不是我做的** ——
三个文件的 mtime 都是 `07:59`，正是新作业启动、另一位执行者把补丁应用到 `repo/SuGaR` 上
以便单卡重跑 λ 组的时刻；我本次会话从 08:00 起只在 `scripts/deliverables/`、`repo_release/`、
`outputs/`、`notes/` 下写文件，对 `repo/` 只有读操作（读过 `repo/SuGaR_dev/sugar_utils/dnc_utils.py`
和 `notes/dnc_stage3.patch`）。`git log -1` 仍是官方 commit `7c10c4ae…`，没有任何 commit。

- 新增文件：`scripts/deliverables/{_deliv_common.py,make_figures.py,make_report.py}`、
  `repo_release/{README_ZH.md,.gitignore,scripts/reproduce_all.sh}`、
  `outputs/figures/**`、`outputs/reports/report*.{docx,pdf,json}`、`outputs/previews/report*/**`、
  `notes/doc_toolchain/SELF_CHECK_report*.md`、本文件。
- **未删除任何文件**（包括被我的新命名取代的 `outputs/reports/report_布局验证_baseline.*`）。
- 未 pip install、未动 ML venv、未 git commit、未 push、未使用 GPU。

---

## 7. [DEVIATION] 与需要主会话知情的事项

### [DEVIATION-1] LibreOffice 在默认 locale 下无法加载文件名含中文的文档

- **现象**：`scripts/deliverables/libreoffice.sh --convert-to pdf outputs/reports/report_布局验证_baseline.docx`
  报 `Error: source file could not be loaded`；同一个文件改成 ASCII 文件名后立刻成功。
- **定位**：逐个 locale 实测 ——
  `LC_ALL=C.UTF-8` 成功；`LC_ALL=en_US.UTF-8`（本 shell 默认）失败；`LC_ALL=zh_CN.UTF-8` 失败。
  连 codex 留下的样例 `outputs/reports/全中文科研交付工具链测试报告.docx` 在当前 locale 下也转不了，
  说明这是**环境 locale 问题**，不是我的文件损坏（zip 完整、python-docx 可正常打开）。
- **影响**：`CLAUDE.md §12.1` 给的转 PDF 命令在当前 shell 下对中文文件名会失败。
- **我的处理**：① 不修改 `scripts/deliverables/libreoffice.sh`（属于既有模块，按规则先报告不擅自改）；
  ② 正式交付物一律用 ASCII 文件名（`report.docx`）；③ 转换命令统一加前缀 `LC_ALL=C.UTF-8`。
- **建议**：若主会话同意，可在 `libreoffice.sh` 里加一行 `export LC_ALL=${LC_ALL:-C.UTF-8}` 一劳永逸。

### [DEVIATION-2] 正式骨架 `report.docx` 目前只有 λ=0 一组有指标

λ=0.05 / λ=0.2 的评测文件还没生成（重跑中）。报告里对应的表显示「未完成」、图 7 显示「尚未生成」。
这是**预期的骨架状态**，等三组评测齐了重跑 §1 的两条命令即可，无需改脚本。

### [FINDING] λ=0.2 在首轮训练中出现几何退化（重要，建议主会话重点关注）

首轮被中断的 `coarse_dnc02_killed_0649/dnc_vis/iter012000_*.png` 显示：**λ=0.2 组的渲染深度图已经
退化成一张没有任何场景结构的光滑曲面**，渲染法向图变成一片近乎均匀的绿色，有效像素掩码几乎全白
（`dnc_log.csv` 里 valid_pixel_ratio 从 0.87 升到 0.988，L_dnc 从 0.32 掉到 0.05）；
同一迭代的 λ=0.05 组仍然清晰保留着卡车的结构。

机理推测（**待实验验证，我没有做消融**）：L_dnc 的梯度同时流经渲染法向 N 与渲染深度 D，
「把深度整体抹平」是这个损失的一个平凡解；λ 足够大时优化器会走向这个解。
证据图已经备好：`outputs/figures/_首轮被中断记录/fig5b_DNC中间量_各λ对比.png`（λ=0.05 vs λ=0.2，同为第 12000 迭代）。
报告 §6 已经为此预留了占位符 9（λ 敏感性），并在 §6.1 局限里写了「L_DNC 梯度同时流经 D，存在把深度抹平的平凡解」。

> 注意：这一条基于**首轮被中断的运行**。重跑的 λ=0.2 是否复现同一现象，要等重跑结束后看
> `outputs/runs/coarse_dnc02/dnc_vis/` 与最终指标，**不能直接把首轮观察当成正式结论**。

### [NOTE] 我自己写的代码里发现并当场修掉的 bug

`make_figures.py` 的 fig2 里写成 `gt = gt or _img(...)`，当第一个 run 有 GT 时凑巧能跑，
换成「第一个 run 没有 GT」的组合就会抛 `ValueError: The truth value of an array with more than one
element is ambiguous`。已改成显式 `if gt is None:` 并加注释。这是本任务新写的文件，不属于既有模块。

### [NOTE] 超出要求新增的内容

1. `fig5b`（各 λ 的 DNC 中间量对比）——因为它是 λ=0.2 退化现象最直接的证据；有 ≥2 组 dnc_vis 时自动生成，否则跳过。
2. `figures_manifest.json` 增量合并——用 `--only` 重画部分图时不会把其它图的记录抹掉。
3. 图与表**编号固定**：缺图也占一个编号并插入红色提示框，所以正文里的「见图 N / 见表 N」永远不会错位；
   若顺序被改动，脚本会打印 `[WARN] 图/表数量与交叉引用常量不一致`。
