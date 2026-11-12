# AGENTS.md — PIDNet 实验与协作规范

本文件适用于本仓库内的所有 AI agent（主会话、子代理、外部工具）。开始任何代码修改、训练或评估前必须先读本文件。优先级：**用户当次明确指令 > 本文件 > 通用默认习惯**。

## 1. 项目与环境

- 项目：PIDNet（CVPR 2023）实时语义分割，支持 Cityscapes（19 类）与 CamVid（11 类）。
- 入口：`tools/train.py`、`tools/eval.py`；配置 `configs/`；模型 `models/`；数据 `data/`；权重 `pretrained_models/`。
- **运行环境不固定**：每次开始工作前自行探索并确认——可用的 Python/conda 环境、torch 与 CUDA、GPU 数量与显存、数据与权重是否就绪。不要假设某个环境、GPU 数量或 batch 一定可用；`configs/` 里的 `GPUS` 和 batch 只是默认值，跑之前必须核对。
- 数据集列表：`data/list/cityscapes/*.lst`、`data/list/camvid/*.lst`。测试集不得参与调参、选 checkpoint 或阈值选择。

## 2. 硬性规则（必须遵守）

1. **所有实验必须在 `test/` 下进行。** 禁止在仓库根目录、`output/`、`log/` 或 `/tmp` 里跑正式实验；禁止把产物写进别人的实验目录。
2. **一个实验一个目录**：`test/<YYYYMMDD>-<slug>/`，slug 使用小写字母、数字、短横线，例如 `test/20261111-pidnet-s-baseline/`。
3. **实验开始前必须先写 `plan.md`。** 没有 `plan.md` 不得启动训练、评估、数据生成或任何长时间任务。纯阅读、单次前向烟测可以不做完整 plan，但要在 `notes.md` 记录。
4. **实验结束后必须写 `report.md`**（同一实验目录）。只跑不写报告视为实验未完成。
5. **`plan.md` 和 `report.md` 必须注明作者 agent**，格式见第 4 节。子代理必须署自己的名字，不得冒用主会话或其他 agent。
6. **不要修改 `configs/` 下的基线配置**。需要改配置时，复制到 `test/<exp>/configs/` 或用命令行覆盖；原始配置保持可复现。
7. **不要覆盖或删除他人的实验目录**；新实验开新目录。不要用 `git checkout` / `git reset` 清掉别人未提交的实验改动。
8. **数据与权重只读**：不要修改 `data/` 内容，不要覆盖 `pretrained_models/`。数据集、checkpoint、日志、TensorBoard、预测图不要提交进 git。
9. **实验性代码改动必须可追溯**：任何对 `models/`、`utils/`、`datasets/`、`tools/` 的修改，都要在 `plan.md` 写明改了什么、为什么；在 `report.md` 附 `git diff --stat` 或 commit hash。未经用户确认，不要把实验性改动合入基线。
10. **报告数字必须可复现**：给出精确命令、配置文件、checkpoint 路径、seed、GPU 和评估 split。区分「本地实测」「文献报告」「待验假设」，不要把文献数字写成自己的结果。

## 3. 实验目录结构

```
test/<YYYYMMDD>-<slug>/
├── plan.md          # 必须，实验开始前写
├── report.md        # 必须，实验结束后写
├── configs/         # 本实验使用的配置副本（如有改动）
├── scripts/         # 本实验专用脚本（数据处理、分析、绘图等）
├── logs/            # 训练/评估日志（LOG_DIR）
├── output/          # checkpoint、预测结果（OUTPUT_DIR）
└── notes.md         # 可选：过程记录、失败尝试、临时观察
```

训练/评估输出必须重定向到实验目录：

- `OUTPUT_DIR test/<exp>/output`
- `LOG_DIR test/<exp>/logs`

实际日志会落在 `OUTPUT_DIR/<dataset>/<cfg_name>/` 和 `LOG_DIR/<dataset>/<model>/<cfg_name>_<time>/`，这是正常行为。

## 4. 作者署名规范（plan.md / report.md 必填）

每份 `plan.md` 和 `report.md` 顶部必须有署名块：

```markdown
> **Agent:** <agent 名称>（<模型/后端，如已知>）
> **Date:** YYYY-MM-DD
> **Experiment:** <YYYYMMDD-slug>
```

示例：

```markdown
> **Agent:** GLM-FLASH-WORKER（GLM-5.3-Flash）
> **Date:** 2026-11-11
> **Experiment:** 20261111-pidnet-s-baseline
```

多 agent 合作时逐条列出并注明分工：

```markdown
> **Agents:**
> - ZCode（opencode-go-chat/deepseek-v4.1-flash）— 实验设计、报告撰写
> - GLM-FLASH-WORKER（GLM-5.3-Flash）— 代码实现、训练执行
```

要求：

- 谁写的谁署名；不要只写「AI」「assistant」「agent」这类无法追溯的称呼。
- 子代理完成部分工作后，如果由主会话汇总，主会话必须在署名区列出所有实际参与者及分工。
- 后续有人修改了 plan/report，在文末追加修订记录（谁、何时、改了什么），不要抹掉原作者。

## 5. 工作流

1. 阅读本文件、相关代码和配置，确认数据与权重是否就绪。
2. 创建实验目录，先写 `plan.md`（含署名块），再执行任何训练/评估。
3. 执行前自检：命令可复制执行、输出路径在 `test/` 下、GPU/batch 与本机匹配。
4. 执行时把 stdout/stderr 同时写入 `logs/`（例如 `2>&1 | tee logs/train.log`）。
5. 实验结束后写 `report.md`（含署名块），给出结果表、异常、产物路径和下一步。
6. 汇报时给出实验目录、`plan.md` / `report.md` 的绝对路径和关键结论。

## 6. plan.md 模板

````markdown
# <实验标题>

> **Agent:** <agent 名称>（<模型/后端>）
> **Date:** YYYY-MM-DD
> **Experiment:** <YYYYMMDD-slug>

## 目标与假设
- 要验证什么？预期结果是什么？

## 基线
- 基线配置 / checkpoint / 已有结果及来源。

## 改动点
- 代码文件与改动摘要；配置项差异；不改哪些变量。

## 数据与划分
- 数据集、train/val/test 名单、是否使用额外数据。

## 运行命令
```bash
# 精确、可复制；注明实际使用的环境、GPU、seed、输出路径
```

## 评估指标与成功标准
- 主指标、次指标、对照方式、判定阈值。

## 资源与时间预算
- 预计显存、batch、epoch/步数、预计耗时。

## 风险与回滚
- 可能的 OOM、数据缺失、结果不可比；失败时如何回滚。
````

## 7. report.md 模板

```markdown
# <实验标题> — 结果报告

> **Agent:** <agent 名称>（<模型/后端>）
> **Date:** YYYY-MM-DD
> **Experiment:** <YYYYMMDD-slug>

## 结论摘要
- 3–5 句话说明是否支持假设。

## 实验设置
- git commit / 分支、配置文件、seed、GPU、batch、训练步数。
- 实际执行的命令（与 plan 不一致时注明差异）。

## 结果
| 指标 | 基线 | 本实验 | 差值 |
| --- | --- | --- | --- |
| mIoU | | | |

## 异常与失败
- OOM、数据缺失、指标异常、被放弃的尝试。

## 产物路径
- checkpoint、日志、TensorBoard、预测结果。

## 下一步
- 是否复验、需要哪些对照、是否值得继续。
```

## 8. 常用命令

```bash
cd /home/robot/catkin_ws/PIDNET

# 先自行探索并激活可用的 Python 环境，确认 torch/CUDA、GPU 数量与显存
EXP=test/20261111-pidnet-s-baseline
mkdir -p "$EXP"/{configs,scripts,logs,output}

# 训练（GPUS 和 batch 是示例值，必须按实际环境调整）
python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
  GPUS "(0,)" TRAIN.BATCH_SIZE_PER_GPU 2 \
  OUTPUT_DIR "$EXP/output" LOG_DIR "$EXP/logs" \
  2>&1 | tee "$EXP/logs/train.log"

# 评估（checkpoint 路径按实际输出调整）
python tools/eval.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml \
  TEST.MODEL_FILE "$EXP/output/cityscapes/pidnet_small_cityscapes/checkpoint.pth.tar" \
  OUTPUT_DIR "$EXP/output" LOG_DIR "$EXP/logs" \
  2>&1 | tee "$EXP/logs/eval.log"
```

说明：

- yacs 会把 `(0,)` 解析成 tuple；`OUTPUT_DIR` / `LOG_DIR` / `TEST.MODEL_FILE` 用普通字符串即可。
- `tools/train.py` 会检查 `torch.cuda.device_count()` 与 `GPUS` 数量是否一致，不一致会直接退出。
- 训练默认每轮在 `OUTPUT_DIR/<dataset>/<cfg_name>/` 下保存 `checkpoint.pth.tar`。
- 评估默认读 `best.pt`，所以通常必须显式传 `TEST.MODEL_FILE`。

## 9. 可复现性清单

每次实验至少记录：

- `git rev-parse HEAD` 和 `git status --short`
- 解析后的完整配置（训练脚本会打印；建议同时复制 yaml 到实验目录）
- seed（默认 304）、GPU 型号与数量、每卡 batch、总步数/epoch
- 数据名单文件及是否改动过预处理
- 评估 split（val / test）和是否使用 flip / multi-scale / TTA
- checkpoint 的绝对路径与文件大小/哈希（可选）

## 10. 维护记录

- 2026-11-11：由 ZCode 初始化本文件，建立 `test/` 实验目录规范。
