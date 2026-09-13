# rclone → Google Drive 交付通道（2026-09-13 实测）

## 环境事实
| 项 | 值 |
|---|---|
| rclone 二进制 | `/home/svu/e1351071/bin/rclone`（已在 PATH） |
| 版本 | v1.73.4 |
| 配置文件 | `~/.config/rclone/rclone.conf` |
| remote 名 | **`googleDrive`**（type=drive, scope=drive, 个人盘非团队盘） |
| Drive 容量 | 5 TiB，已用 801.8 GiB，**剩余 4.2 TiB** |
| token | 配置里的 access_token 已于 2026-07-24 过期，但 refresh_token 有效，rclone 自动续期成功 |

## 实测结果
- `rclone about googleDrive:` ✅ 通
- 小文件上传 `googleDrive:/zju_test/` ✅ 成功，云端回读内容一致
- 100 MB 测速：**总耗时 115 s**，其中第一次传输速率从 5 MiB/s 一路衰减到 30 KiB/s 卡死，
  rclone 自动重试后第二次很快传完（实际传了 200 MiB 才成功一次）
  → **有效吞吐 ≈ 0.9 MB/s（含重传），链路不稳定，必须带重试参数**
  → 交付脚本已加 `--retries 5 --low-level-retries 20 --contimeout 30s --timeout 5m`

## 交付脚本
`$PROJ_ROOT/upload_to_gdrive.sh`（已 chmod +x，扫描模式验证通过）

**核心流程：上传前强制先扫描分类，人工确认后才执行**（用户 2026-09-13 明确要求）

```bash
./upload_to_gdrive.sh --scan     # 只扫描不上传，随时可跑，退出码 0
./upload_to_gdrive.sh            # 扫描 -> 打印清单 -> 输入 yes -> 才真正上传
./upload_to_gdrive.sh --yes      # 跳过交互确认（非交互环境用，如 nohup 后台）
./upload_to_gdrive.sh --with-model ckpts/my_lora   # 额外上传指定权重
```

扫描报告含四段：
1. 【✔ 将上传】逐个文件 + 大小（按大小降序）
2. 【✘ 将排除】按顶层目录汇总，非预期目录会标 `<- 非预期`
3. 【⚠ 需人工过目】>50MB 的待传文件；被排除但**不在已知排除目录内**的文档/代码类文件（疑似误伤）
4. 预估上传耗时

非交互环境（stdin 非 tty）不加 `--yes` 会拒绝上传并 exit 2，防止误触。

### 排除规则（--scan 已验证生效）
| 规则 | 原因 |
|---|---|
| `/ckpts/**` | 模型权重，需要时用 `--with-model` 单独传 |
| `/data/**` | 数据集 |
| `/wheels/**` | 手动下载的 .whl |
| `/repo.baseline/**` | 魔改前的代码备份副本 |
| `/tools/**` | **codex 装的 skill venv(doc_env)**，已涨到 28257 文件/1.6GB，含 symlink 不可移植 |
| `/assets/**` | **codex 装的 Noto CJK 字体**，79.6MB，仅本地排版用 |
| `**/__pycache__/**` `**/*.pyc` `**/.ipynb_checkpoints/**` `**/.git/lfs/**` | 缓存垃圾 |
| `**/*.safetensors *.ckpt *.pth *.pt *.onnx *.msgpack *.gguf` | 兜底，防权重躲在别的目录 |

保留上传：`CLAUDE.md`、`plans/`、`notes/`、`logs/`、`repo/` 源码、`outputs/*.mp4`、`.agents/`(72K)

用 `copy` 而非 `sync` —— 不会删除云端已有文件（符合"删除前先问"规则）。

### ⚠️ 踩过的坑
- `tools/` 和 `assets/` 是 **codex 的 scientific-deliverables skill 自动装的**，会持续增长
  （8677 → 15763 → 28257 文件），每次交付前跑 `--scan` 确认没有新的类似目录冒出来。
- Google Drive 的瓶颈是 **API 调用次数而非带宽**，几千个小文件比一个大文件慢得多。

## 时间估算（交付阶段排期用）
按 ~1 MB/s 保守估：代码+日志+若干 mp4 若共 200 MB → **约 3–4 分钟**。
若要额外传 10 GB 权重 → **约 3 小时，来不及**，需提前单独后台起。
