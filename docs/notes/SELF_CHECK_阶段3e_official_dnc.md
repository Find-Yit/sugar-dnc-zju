# SELF_CHECK — 阶段 3e：官方 SuGaR dn_consistency 对照组 `coarse_official_dnc`

> 执行者 8（opus-executor）。本文件在运行过程中分段写入，**最终状态见文末「11. 总结论」**。
> 硬性规则遵守情况：未删除任何文件、未 `git commit`、未 `pip install`、未修改任何已有文件（只新增 3 个文件）、
> 全程独占 GPU0（等待前序任务信号后才启动）。

---

## 1. 目的

官方 SuGaR 仓库自带一份 2DGS 式的深度-法向一致性正则
（`sugar_trainers/coarse_density_and_dn_consistency.py`），与本项目自研的 DNC（`sugar_utils/dnc_utils.py` +
`sugar_trainers/coarse_density.py` 补丁）思路重合。官方版本只能通过 `train.py -r dn_consistency` 调用，
而那是「coarse + refine + 贴图网格」的全流程脚本，本次考核不跑 refine，因此**需要一个只跑 coarse 的入口**，
才能在同一场景、同一 3DGS 7k 权重、同一迭代预算、同一评测协议下与自研 DNC 正面对比。

---

## 2. 新增文件（只增不改）

| 文件 | 作用 | md5 |
|---|---|---|
| `repo/SuGaR_dev/train_coarse_official_dnc.py` | 官方 dn_consistency coarse 训练入口（本次实际运行） | `ff04346a57b2f6be54be980956c79478` |
| `repo/SuGaR/train_coarse_official_dnc.py` | 同一文件的副本（分支 `dnc`，**未 commit**，留作仓库交付用） | `ff04346a57b2f6be54be980956c79478` |
| `scripts/run_s3e_official_dnc.sh` | 等待 GPU 信号 → 训练 → mesh 提取 → 评测的串联脚本 | — |

启动前后两个仓库的 `git status --short`（证明没有改动任何已有文件、也没有 commit）：

```
# repo/SuGaR (branch dnc @ 7c10c4a)
 M sugar_trainers/coarse_density.py      <- 阶段 3 的 DNC 补丁（他人所为，我未触碰）
 M train_coarse_density.py               <- 同上
?? sugar_utils/dnc_utils.py              <- 同上
?? train_coarse_official_dnc.py          <- 本次新增

# repo/SuGaR_dev (branch main @ 7c10c4a)
 M sugar_trainers/coarse_density.py
 A sugar_utils/dnc_utils.py
 M train_coarse_density.py
?? train_coarse_official_dnc.py          <- 本次新增
```

`sugar_trainers/coarse_density_and_dn_consistency.py` 在两个仓库里 `diff` 结果为**完全一致且未被修改**
（不在 `git status` 中出现），即官方正则代码一行未动。

---

## 3. 官方 DNC 与自研 DNC 的实现差异（用于解释指标差异）

| 维度 | 官方 `coarse_density_and_dn_consistency.py` | 自研 `dnc_utils.py` + `coarse_density.py` |
|---|---|---|
| 损失式 | `mean(1 - <N_view , N_d_view>)`，**全图像素求平均** | `mean(1 - \|<N, N_d>\|)`，**只在有效像素上平均** |
| 绝对值 | 无（依赖法向已被显式翻向相机） | 有（消除 3D 椭球最短轴的朝向歧义） |
| 有效像素掩码 | **无**（背景/未命中像素也参与） | `D < 0.98·max_depth`、去 2 像素边界、深度相对梯度 < 0.05、`‖N_raw‖ > 0.1` |
| 法向来源 | `sugar.render_depth_and_normal()`：逐高斯最短轴 → 按 `sign((n·(c−p)))` 翻向相机 → 渲染 view 空间 x,y 分量 → `z = −sqrt(1−x²−y²)` | 逐高斯最短轴 → 翻向相机 → 渲染三分量 → 像素上再归一化 |
| 深度→法向 | 2DGS 原版 `depths_to_points` + `depth2normal_2dgs`（COLMAP 相机、world 空间叉乘后转 view 空间），中心差分 | PyTorch3D NDC 解析反投影 + 中心差分叉乘，全程 view 空间 |
| 梯度路径 | 同时经深度 D 与法向 N（无 detach） | 默认同官方；`--dnc_detach_depth True` 时只经 N（阶段 3d 新增） |
| 权重 λ | `dn_consistency_factor = 0.05`（源码硬编码） | 命令行 `--dnc_factor`，本次对照取 0 / 0.05 / 0.2 |
| 起始迭代 | `start_dn_consistency_from = 9000`（硬编码），条件 `iteration > 9000` | `--dnc_start 9000`，同条件 |

> 结论性提示：**官方版本的 λ=0.05 与自研 `coarse_dnc005` 是最接近的一对**，两者主要差在
> 「有无有效像素掩码」「有无绝对值」「深度→法向的具体离散化」三点。

---

## 4. 新入口脚本做了什么 / 没做什么

`train_coarse_official_dnc.py` 是**薄封装**：

- 参数与 `train_coarse_density.py` 完全一致（`-s -c -o -i --eval --white_background -e -n --gpu`），另加 `--seed`（默认 0）。
- 调用前 `random.seed / np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all`（与自研训练器同一写法、同一时机）。
- 用 `time.time()` 与 `torch.cuda.max_memory_allocated()` 在调用前后测墙钟与峰值显存（调用前先 `reset_peak_memory_stats`）。
- 训练结束后从返回的 `model_path`（`.pt`）读 `state_dict['_points'].shape[0]` 统计高斯数。
- 写 `<output_dir>/train_stats.json`，字段名与自研 run 对齐（`train_wallclock_min`、`mean_time_per_iteration_ms`、
  `max_memory_allocated_MiB`、`n_gaussians_final`、`seed`），另加 `regularization='official_dn_consistency'`、
  `dn_consistency_factor=0.05`、`start=9000`。
- **不改官方训练器一行代码**，不改任何超参（λ 与起始迭代都由官方源码硬编码）。

---

## 5. CPU 侧自检（GPU 空出前完成）

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法编译（dev） | `python -m py_compile train_coarse_official_dnc.py`（在 `repo/SuGaR_dev`） | PASS |
| 语法编译（SuGaR） | 同上（在 `repo/SuGaR`） | PASS |
| `--help` 真导入 | `python train_coarse_official_dnc.py --help` | PASS，退出码 0，10 个参数齐全 |
| argparse 解析值 | 打桩脚本（stub 掉官方训练器与全部 CUDA 调用，**不创建 GPU 上下文**）跑真实命令行 | PASS：`ARGPARSE_TEST_OK`，10 个参数逐个与预期一致，`-c` 以 `/` 结尾 |
| 高斯计数函数 | 对已有 `coarse_base_seed0/15000.pt` 调 `_count_gaussians` | PASS：得 429423，与该 run 的 `train_stats.json` 的 `n_gaussians_final` 逐位一致 |
| 串联脚本语法 | `bash -n scripts/run_s3e_official_dnc.sh` | PASS |
| 等待逻辑（超时分支） | `DEADLINE=0001 bash scripts/run_s3e_official_dnc.sh` 干跑 | PASS：打印 "DEADLINE 0001 reached … NOT launching"，退出码 3，**未碰 GPU** |
| 信号 grep 逻辑 | 用 `run_s34_one.sh` 的原样格式伪造一行到临时文件 | PASS：`SIGNAL_GREP_OK` |

打桩脚本：`<scratchpad>/test_parse_official_dnc.py`（不落入项目目录，属临时验证物）。

---

## 6. GPU 排队与实际启动

只有一张 H200。按主会话要求，脚本先阻塞等待执行者 7 的 `coarse_dnc02_detach` 链打印
`MESH (coarse_dnc02_detach) finished`（在 `logs/s34_coarse_dnc02_detach.log` 或 `logs/s34_mesh_dnc02_detach.log`），
截止 09:35；超时则**不启动**并退出码 3。等待期间额外确认本用户没有 `train_coarse_density.py` /
`extract_mesh.py` / `train_coarse_official_dnc.py` 进程仍在跑，确保不与他人并发占卡。

- 脚本启动（进入等待）：`2026-09-13 08:34:34`，PID 1339868，等待日志 `logs/s3e_official_dnc_waiter.log`。

（时间线与结果见下节，运行结束后填写。）

---
