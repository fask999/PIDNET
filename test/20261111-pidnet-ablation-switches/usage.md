# PIDNet 消融开关

训练、评估和预测读取同一套 `ABLATION` 配置。原 `configs/` 文件可继续使用，不加这些字段时走原模型和旧边界目标。

| 配置项 | 默认值 | 用法 |
| --- | --- | --- |
| `ABLATION.CONTEXT` | `auto` | S/M 用 PAPPM，L 用 DAPPM。可显式选 `pappm`、`dappm`、`pmpm`、`denseaspp`。 |
| `ABLATION.DENSE_RATES` | `[1,2,3,4]` | DenseASPP 四层空洞率，四个正整数；次序即串接次序。 |
| `ABLATION.BOUNDARY.TARGET` | `legacy` | 原先的 Canny；`valid` 改为缩放、裁剪和翻转后生成的有效四邻接边界。 |
| `ABLATION.BOUNDARY.WIDTH` | `0` | 自动取 legacy 的 4×4 核，或 valid 的 3×3 核。valid 显式使用正奇数；1 只标原始界面两侧。 |
| `ABLATION.BOUNDARY.MASK_VOID` / `VOID_RADIUS` | `false` / `0` | 开启后，从边界 BCE、Dice 和 BAS 中排除 void/pad；半径以增强后的像素为单位，0 只排除本身，正数额外排除方形邻域。 |
| `ABLATION.BOUNDARY.DICE_WEIGHT` | `0.0` | 加在原 20 倍平衡 BCE 上的 soft Dice 系数，例如 `1.0`。可以独立于目标和 mask 切换。 |
| `ABLATION.I_ATROUS.MODE` | `none` | `fixed` 固定空洞，`fadc` 空间自适应空洞。 |
| `ABLATION.I_ATROUS.STAGE` / `BLOCK` / `RATE` | `5` / `1` / `2` | 只改 I 路选中块的 3×3 `conv2`，不碰首个下采样块。块号从 0 数；stage 3/4 的 S/M 可选 1、2，L 可选 1、2、3；stage 5 只选 1。fixed 的 rate 1 是原卷积对照，FADC 的 rate 是初始化标称值。 |

新边界目标不会用 void/pad 产生正例。单开 `valid` 时 void/pad 仍可作为负例；要排除其监督须同时开 `MASK_VOID`。单开 legacy mask 仍保留有效区的旧 Canny 目标，近 void 的旧伪边缘可用邻域排除；用 valid 才从源头排除 void 界面。

`fadc` 需要与 PyTorch 和目标设备匹配、带编译 `deform_conv2d` 的 torchvision。已验证 PyTorch 1.13.0、torchvision 0.14.0 和 CUDA 11.7；其余模式不加载 torchvision 的该算子。

```bash
cd /home/robot/catkin_ws/PIDNET
EXP=test/20261111-pidnet-ablation-switches
export PIDNET_PYTHON=/home/robot/anaconda3/envs/segformer/bin/python

# 查看各组精确命令，不启动任务
bash "$EXP/scripts/run_ablation.sh" show cityscapes baseline
bash "$EXP/scripts/run_ablation.sh" show cityscapes pmpm
bash "$EXP/scripts/run_ablation.sh" show cityscapes denseaspp
bash "$EXP/scripts/run_ablation.sh" show cityscapes valid-edge
bash "$EXP/scripts/run_ablation.sh" show cityscapes void-mask ABLATION.BOUNDARY.VOID_RADIUS 1
bash "$EXP/scripts/run_ablation.sh" show cityscapes dice
bash "$EXP/scripts/run_ablation.sh" show cityscapes boundary
bash "$EXP/scripts/run_ablation.sh" show cityscapes fixed1
bash "$EXP/scripts/run_ablation.sh" show cityscapes fixed2
bash "$EXP/scripts/run_ablation.sh" show cityscapes fadc
bash "$EXP/scripts/run_ablation.sh" show cityscapes combo-fadc
bash "$EXP/scripts/run_ablation.sh" show camvid combo-fixed
```

数据和同源初始化权重就绪后，把 `show` 换成 `train`；用相同数据集、预设和覆盖项改成 `eval` 即在验证集评估。用 `PIDNET_RUN_TAG=seed304-rep2` 等区分重跑，并在训练和评估保持一致。首次运行先按可用 GPU 统一校准所有对照组的 `GPUS`、crop 和 batch，训练时每卡至少 2 个样本以通过原池化 BN。复制配置默认分别用 Cityscapes-M 19 类、CamVid-S 11 类的 `train.lst` → `val.lst`，训练 seed 304，评估单尺度不 flip；正式结果只在相同配方内比较。显式 `MODEL.PRETRAINED ''` 可做从零初始化的对照。

例如 Cityscapes `combo-fadc` 的默认输出为 `$EXP/output/cityscapes/combo-fadc/cityscapes/pidnet_medium_cityscapes/`；`eval` 默认读其 `best.pt`，也可用 `TEST.MODEL_FILE` 显式指定训练保存的 `checkpoint.pth.tar`。启用空洞时，权重记录模式、stage、block 和率；恢复训练、评估和预测须使用相同设置，错配会明确报错。训练预训练仍允许加载部分兼容权重，不会改变目标模型的设置。

```bash
CKPT="$EXP/output/cityscapes/combo-fadc/cityscapes/pidnet_medium_cityscapes/best.pt"
"$PIDNET_PYTHON" -B tools/custom.py \
  --cfg "$EXP/configs/pidnet_medium_cityscapes.yaml" --p "$CKPT" --r samples \
  --output-dir "$EXP/output/cityscapes/combo-fadc/predictions" \
  ABLATION.CONTEXT denseaspp ABLATION.I_ATROUS.MODE fadc
```

示例预测图使用现有 Cityscapes 调色板。边界监督配置不参与预测模型结构。所有正式精度、边界指标和延迟仍需在相同数据、预处理、初始化和计时口径下实测。
