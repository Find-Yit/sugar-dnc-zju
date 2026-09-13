# report.tex —— Overleaf 编译说明（由 report.docx 自动转换，本机未编译验证）

本目录是 `outputs/reports/report.docx`（PDF 版 34 页）的 LaTeX 等价工程，文字、数字、表格与图片顺序与 docx/PDF 一致，未做任何改写或重算。

**使用步骤（Overleaf）**：登录 Overleaf → `New Project` → `Upload Project` → 把本目录（`report.tex` + `figures/`，约 51 MB）打包成 zip 上传（或新建空项目后把 `report.tex` 与整个 `figures/` 文件夹一起拖入，保持 `figures/` 目录结构）→ 打开 `Menu → Settings → Compiler` 选择 **XeLaTeX**（必须，中文用 `ctexart` + xeCJK）→ 点 `Recompile`。文档用 `longtable` 排表，通常需要编译 **2 次**表格与交叉引用才稳定。

**已做的语法级自检**（脚本扫描，非真实编译）：`\begin`/`\end` 环境配对一致（document 1 / center 2 / itemize 10 / longtable 16 / verbatim 7 / figure 12）；花括号总深度归零；每行 `$` 数目成对；12 处 `\includegraphics` 引用的文件全部存在且文件名为纯 ASCII；正文（verbatim 之外）无未转义的 `% _ # $ & { } ~ ^ \`。

**注意**：本机（hopper-29 计算节点）没有 `xelatex` / `pdflatex` / `tectonic`，因此**未做真实编译验证**，只做了上述语法级检查。附录里的命令块用 `verbatim` 排版、不自动折行，最长一行 165 字符（已按最长行自动调小字号到 `\tiny`/`\scriptsize`），若在 Overleaf 中仍有 overfull hbox 溢出页边，手工换行即可（不会报错，只是警告）。`fig2_render_compare.png`（22 MB）与 `fig3_mesh_normal.png`（11 MB）体积较大，上传慢时可用同名 PDF 矢量版（`outputs/figures/fig2_渲染对比.pdf` 等）替换。

**转换脚本**：`scripts/deliverables/docx2tex_report.py`（用 `tools/doc_env/bin/python` 运行，直接读 `outputs/reports/report.docx` 重新生成本目录，图片按 md5 与 `outputs/figures/`、`outputs/a100/results_a100/figures/` 内的源 PNG 匹配后改成 ASCII 文件名复制进 `figures/`）。
