# PIDNet-L 单卡 RTX 4090 训练启动器 — 使用说明

> **Agent:** deepseek-writer（opencode-go/deepseek-v4.1-flash / high）
> **Date:** 2026-11-12
> **Ticket:** 338146450247493f9b622188a8549386；writing 单元

本文件说明 `tools/train_pidnet_l_4090.py` 的用法与边界。文中的命令除短测命令外都是**示例，未启动**；本文件不包含训练、精度、耗时或显存结论。实测证据与复现性预审见 [开发报告](../test/20261112-pidnet-l-4090-launcher/report.md)。

## 1. 入口与依赖

启动器在仓库根运行，负责 GPU 选择、只读预览、输入与权重预检、逐组实验目录和运行记录，并在预检通过后以 FP32 调用 `tools/train.py --cfg <run>/configs/resolved.yaml --seed 304`。训练入口本身按配置做验证；启动器不解析或汇总 mIoU。

它复用仓库内的旧 L 套件，不能单独复制一个文件使用：

- 旧 runner 与配置：`test/20261112-pidnet-l-ablation/scripts/run_ablation.py`、`test/20261112-pidnet-l-ablation/configs/`，模型开关来自 `utils/ablation_config.py`。
- 运行记录模板：`test/20261112-pidnet-l-4090-launcher/templates/{plan,report}.md`，缺失或被改写会影响生成的 `plan.md` / `report.md`。

## 2. 环境

本机复现例只用 `segformer` 环境（`conda activate segformer`，再用该环境里的 `python`）。该环境实测组合（`inputs.json`）：`/home/robot/anaconda3/envs/segformer/bin/python`，Python 3.7.16、torch 1.13.0、torchvision 0.14.0、torch CUDA build 11.7、numpy 1.21.6、cv2 4.1.2、PIL 9.4.0、tqdm 4.65.2。真实 4090 机器需自行准备可用的 CUDA PyTorch 及与其匹配的 torchvision；FADC 预设还要求该 torchvision 提供可用的 CUDA `deform_conv2d`。本机未验证任何 4090 版本组合，不提供未经实测的搭配建议。

兼容提示：

- 旧代码使用 `np.int`；numpy 1.21.6 仍提供该别名但发 `DeprecationWarning`。预检只检查它是否存在，不会修改 NumPy。
- FADC 预设（`fadc`、`combo-fadc`）预检会在 CUDA 上跑一次 `torchvision.ops.deform_conv2d` 烟测；本机无 4090，故本启动器的 FADC CUDA 预检未实测。旧 L 实验在 GTX 1650 上做过的合成 FADC 检查属于另一入口，不能替代本预检。
- `--help` 只依赖标准库，可在未装 torch/yacs 的解释器下使用（已在仓库外以 `-S` 验证）。
- `--dry-run` 会导入旧 runner 与配置，因此需要 torch/yacs；它不检查设备、数据或权重，也不写任何文件（含 `__pycache__`）。

## 3. 前置条件

数据根（默认仓库 `data/`，或显式 `--data-root`）须包含 `<root>/list/<dataset>/{train,val}.lst`，每行两列；名单须与仓库 canonical 名单的条目和数量一致（校验按排序后的行集合比较，行序可不同）；名单引用的图像与标签须位于 `<root>/<dataset>/<path>`。Cityscapes 为 19 类、2975/500 行，CamVid 为 11 类、367/101 行。训练只用 `train.lst`，验证只用 `val.lst`，`test.lst` 不进入训练或选型，`EXTRA_TRAIN_SET` 为空。

权重：

- Cityscapes 默认 `pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar`，本机缺失（仅有 readme）；可用 `--pretrained` 指定同 profile 文件。
- CamVid 无默认权重，真实运行必须用 `--pretrained` 显式给出可信的 19 类 Cityscapes-L 权重；预检核对 19 类分类头。
- 预检记录权重 profile（`imagenet-l` / `cityscapes-l`）、绝对路径与 SHA-256。文件存在且能加载，不证明来源或来源训练 split。本机两个文件需分开看（依据 `test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.manifest.json`）：`pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar` 是 **mmseg 原包**（manifest 的 `source.path`，`source_provenance_verified: false`）；`test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.pth.tar` 是由它补零 `conv1.0.bias`、`conv1.3.bias` 得到的**转换产物**（manifest 的 `target.path`）。转换产物的来源与训练 split 未核，两者都不是正式初始化推荐，只适合兼容性烟测。

## 4. 命令

以下示例在仓库根执行，**均未启动训练**。

```bash
cd /home/robot/catkin_ws/PIDNET
conda activate segformer
python -B tools/train_pidnet_l_4090.py --help
```

默认预览（Cityscapes / baseline / batch 2 / 物理 GPU 0 / FP32）：

```bash
cd /home/robot/catkin_ws/PIDNET
conda activate segformer
python -B tools/train_pidnet_l_4090.py --dry-run
```

Cityscapes baseline 实际训练命令（需数据与 ImageNet-L 权重就绪，示例未启动）：

```bash
cd /home/robot/catkin_ws/PIDNET
conda activate segformer
python -B tools/train_pidnet_l_4090.py \
  --dataset cityscapes --preset baseline --run 20261112-pidnet-l-city-baseline \
  --gpu 0 --batch-size 2
```

CamVid 自选预设，显式给出可信 19 类 City-L 权重（路径为占位，示例未启动）：

```bash
cd /home/robot/catkin_ws/PIDNET
conda activate segformer
python -B tools/train_pidnet_l_4090.py \
  --dataset camvid --preset combo-fadc --run 20261112-pidnet-l-cam-combo-fadc \
  --pretrained /abs/path/to/verified-pidnet-l-19class-cityscapes-best.pt \
  --gpu 0 --batch-size 2
```

I 套件（9 组串行；先预览，示例未启动）：

```bash
cd /home/robot/catkin_ws/PIDNET
conda activate segformer
python -B tools/train_pidnet_l_4090.py \
  --dataset cityscapes --i-suite --run 20261112-pidnet-l-i-suite --dry-run
python -B tools/train_pidnet_l_4090.py \
  --dataset cityscapes --i-suite --run 20261112-pidnet-l-i-suite \
  --gpu 0 --batch-size 2
```

`--run` 必须是 `YYYYMMDD-slug`（小写字母、数字、短横线）。省略时默认 `当前日期-pidnet-l-<PID>`。真实运行和预览都会检查目标路径，任一目标已存在即拒绝；套件在运行前检查全部 9 个目录。

## 5. 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | cityscapes | `cityscapes` 或 `camvid` |
| `--preset` | baseline | 14 个旧预设之一，可与 `--i-suite` 互斥 |
| `--i-suite` | 关 | 固定 9 组：baseline、pappm、pmpm、denseaspp、fixed1、fixed2、fadc、combo-fixed、combo-fadc |
| `--run` | 日期-pidnet-l-PID | 单组目录 `test/<run>`，套件为 `test/<run>-<preset>` |
| `--gpu` | 0 | PCI bus 顺序的物理序号 |
| `--batch-size` | 2 | 每卡 batch，必须 ≥2 |
| `--workers` | 配置值（City 6 / Cam 3） | 范围 0–64 |
| `--data-root` | 仓库 `data/` | 自定义数据根 |
| `--pretrained` | 配置值 | CamVid 必填；Cityscapes 可覆盖 |
| `--dry-run` | 关 | 只打印命令与解析配置 |
| `--stage` / `--block` | 5 / 1 | I 分支改动位置；未显式给出时保持旧 L 逻辑（stage 5、block 1、conv2） |

`--block` 选择非首块：stage 3 或 4 可取 1–3，stage 5 只能取 1；越界由配置校验拒绝。`pappm` 是额外对照，`dappm`、`valid-edge`、`void-mask`、`dice`、`boundary` 等其他旧预设可单跑。

## 6. GPU 选择

启动器先设 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 与 `CUDA_VISIBLE_DEVICES=<--gpu>`，再导入 torch 与旧 runner，子进程只见到逻辑 0（`GPUS: (0,)`）。预检要求 CUDA 恰好暴露一张卡，设备名须为桌面版 `GeForce RTX 4090`（允许 `NVIDIA ` 前缀）；`4090D`、`Laptop GPU` 与其他卡都拒绝，且拒绝发生在训练进程创建之前。无 CUDA 时预检直接失败。预览不校验 GPU 是否存在，`--gpu 17` 也可用于在无卡机器上生成计划。

## 7. 目录、记录与退出

每个 run 目录与旧套件同构：

```
test/<run>/            （套件为 test/<run>-<preset>/）
├── plan.md            # 导入旧 runner 与预检之前写入
├── notes.md           # 启动前写入，结束后按实际状态更新
├── configs/resolved.yaml + resolved.sha256
├── logs/train.log
├── output/
└── report.md          # 正常或失败退出后尽量写（磁盘写入异常时不保证生成）
```

目录用独占 `mkdir` 创建，不接管已有路径；`test/` 必须是真实目录。配置冻结且不 resume、不自动调整 batch/epoch/学习率；已保存配置可被旧套件的 `saved_config` 读取并校验 SHA（短测已覆盖），端到端 eval 未实测。

退出语义（shell 看到的是最终结果）：

- 预检或启动失败：stderr 打印 `Preflight or launch failed: ...`；notes/report 尽量写入，磁盘写入异常时不保证生成；返回 1 或 `128+信号`。
- 训练子进程非零：原样保留；被信号终止换算为 `128+信号`。
- 子进程成功但记录写入失败：返回 1；套件随即停止，不跑后续预设。
- 记录失败时 stderr 给出失败路径、异常类型、原始子进程码与当时换算后的 shell 码；notes 与 report 分别尝试写入，两次都失败时不留记录文件。
- 收到 SIGINT/SIGTERM 时转发给子进程组，超时后清理后代。

## 8. 预检内容

按顺序记录并核验：git HEAD、`git status --short`、`git diff --stat`；Python 与解释器；torch / CUDA build；唯一可见 GPU 的名称、物理选择、逻辑 0 与总显存，可用时记录空闲显存；cv2、PIL、tensorboardX、tqdm 可导入；`np.int` 存在；CUDA 基础算子；FADC 预设的 `deform_conv2d`；train/val 名单路径与 SHA-256；权重 profile、路径与 SHA-256；名单行数与引用文件存在性；随后在 CPU 上按旧 L 逻辑载入共享张量做兼容检查。任一项失败即不启动。记录写在 run 目录，不静默安装或下载依赖。

## 9. 训练协议

seed 304、FP32（无 AMP）、单卡、不做梯度累积或自动 LR 缩放。Cityscapes：19 类、1024×1024、484 轮、LR 0.01。CamVid：11 类、960×720、121 轮、LR 0.005。验证沿用 `val.lst`、单尺度、`FLIP_TEST=false`、`MULTI_SCALE=false`；测试集不参与训练、选 checkpoint 或阈值选择。

## 10. 短测与证据

本机复现短测沿用绝对解释器路径：

```bash
cd /home/robot/catkin_ws/PIDNET
/home/robot/anaconda3/envs/segformer/bin/python -B test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py
```

本轮写作未执行该命令。最近证据：worker 终态与 lead 独立复跑各 20 项通过（同一组，不相加），日志 `test/20261112-pidnet-l-4090-launcher/logs/check-launcher-record-failure-verified.log`、`.super-agent/evidence/338146450247493f9b622188a8549386/lead-tests.log`；公开入口与退出码核验见同目录 `lead-entry.log`；环境、名单行数与交集见 `inputs.json`。

## 11. 已知限制

- 本机无桌面 4090，也无本启动器的 FADC CUDA 预检实测；4090 上的显存占用与耗时均未实测，不能由配置推断。模型名称判定只有模拟证据。旧 L 实验在 GTX 1650 上的合成 FADC 检查不属于本启动器预检。
- 本机为 GTX 1650：`nvidia-smi` 显示 4096 MiB，torch 设备属性 3903 MiB（两者来源不同，勿混用）。
- 真实数据与官方权重缺失：名单引用文件均不存在，City 默认 ImageNet-L 权重不存在，Cam 无可信初始化权重。
- 名单键交集实测为 0（train/val、train/test、val/test），但图像缺失，无法核实际像素级泄漏。
- 所有训练、评估、精度与性能数字均未产生；正文数字来自配置、名单文件或短测日志，已在报告中标明来源。
