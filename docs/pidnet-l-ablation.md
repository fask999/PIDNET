# PIDNet-L 消融用法

Cityscapes 和 CamVid 的新配置都使用 PIDNet-L（m=3、n=4、stem 64、上下文分支宽度 112、语义头宽度 256），分别输出 19 和 11 类。模型开关默认关闭：上下文为 DAPPM、融合为 Bag、I 分支保持原始卷积，边界监督沿用旧配置。L 的 PMPM/DenseASPP 是替换 **DAPPM**；`pappm` 是额外对照。早期 M/S runner 与其结果保留，不能当作 L 的结果。

推荐入口：`/home/robot/catkin_ws/PIDNET/test/20261112-pidnet-l-ablation/scripts/run_ablation.py`。

| 比较项 | 预设 | 控制项 |
| --- | --- | --- |
| 原始 L | `baseline`、`dappm` | `baseline` 用 auto 得到 DAPPM；显式 `dappm` 同结构 |
| 上下文 | `pappm`、`pmpm`、`denseaspp` | I 分支卷积、Bag、边界保持默认 |
| I 分支卷积 | `fixed1`、`fixed2`、`fadc` | DAPPM 和默认边界保持相同；分别为固定率 1、2 和 FADC 标称率 2 |
| I 分支组合 | `combo-fixed`、`combo-fadc` | 分别为 PMPM + fixed2、DenseASPP + FADC；边界与 `baseline` 相同 |
| 边界监督 | `valid-edge`、`void-mask`、`dice`、`boundary` | 分别启用有效边界目标、排除 void、Dice（权重 1.0）；`boundary` 启用全部三项，模型结构保持默认 |

固定/FADC 默认改 I stage 5 的第 1 号块的 `conv2`（编号从 0 开始）。也可用 `--stage 3 --block 3` 或 `--stage 4 --block 3` 对比 L 的非首块；改位置时相应对照组也用同一位置。FADC 需要与 PyTorch 和设备匹配的 torchvision 编译版 `deform_conv2d`。

训练与选型使用各自的 `train.lst` → `val.lst`：Cityscapes 2975/500，CamVid 367/101。CamVid 配置将学习率日程设为 121 轮，与现有入口的实际轮数对齐。配置默认单卡、每卡 batch 2；按显存选择统一裁剪尺寸和 batch，同一组对照保持同源权重、seed 304、输入尺寸、训练配方和评估设置。正式选型用单尺度、无 flip 的验证集；方案固定后再另行安排测试集评估。

- **Cityscapes**：默认找 `pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar`，也可用 `--pretrained` 指定同来源文件。检查 stem 和 I 主干，包含能区分同宽 M 的 L 深层块。只加载这组 ImageNet 共享张量。
- **CamVid**：必须用 `--pretrained` 显式指定 19 类 PIDNet-L Cityscapes 权重，例如已核实来源的 L 训练 `best.pt`。检查 19 类语义分类头，再转移 I/P/D 及融合等共享张量，保留上下文、辅助头、类别分类层和 FADC 新参数的当前配置初始化。
- 本机缺真实数据与 L ImageNet 权重。历史 Cityscapes mmseg 原包缺两个 stem bias；其[显式零 bias 转换记录](../test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.manifest.json)仅用于兼容性烟测。原包的训练 split 未核，正式对照请记录权重来源、训练 split 与文件 SHA-256。

从仓库根运行。预览会打印计划命令和完整解析配置，不写入；真实训练前，为每个预设、复跑各自准备新的 `test/YYYYMMDD-slug/plan.md`（含署名），目录中只放该计划。自定义数据根需提供与仓库相同的名单及其引用的图像、标签。

```bash
cd /home/robot/catkin_ws/PIDNET
PY=/home/robot/anaconda3/envs/segformer/bin/python
KIT=test/20261112-pidnet-l-ablation
RUN=20261112-pidnet-l-city-pmpm
"$PY" -B "$KIT/scripts/run_ablation.py" show cityscapes pmpm "$RUN"
"$PY" -B "$KIT/scripts/run_ablation.py" show camvid combo-fadc 20261112-pidnet-l-cam-combo-fadc \
  --pretrained /path/to/verified-pidnet-l-cityscapes-best.pt --gpus 0 --batch-size 2

# 仅在该 run 的计划、实际数据、权重和 GPU 就绪后执行。
"$PY" -B "$KIT/scripts/run_ablation.py" train cityscapes pmpm "$RUN"
CKPT="test/$RUN/output/cityscapes/resolved/best.pt"
"$PY" -B "$KIT/scripts/run_ablation.py" eval "$RUN" --checkpoint "$CKPT"
"$PY" -B "$KIT/scripts/run_ablation.py" custom "$RUN" --checkpoint "$CKPT" --images samples
```

训练把配置、日志和产物分别保存到本次运行的 `configs/`、`logs/` 和 `output/`。评估和预览预测复用 `configs/resolved.yaml`，校验配置 SHA，并要求显式选用该 run 的 `output/` 内 checkpoint。预测图保存在该 run 独立的 `output/custom-*/`；目前 `custom.py` 使用旧的预览配色，正式类别指标使用 `eval`。

直接调用入口时使用同一份配置：`tools/train.py --cfg ...`、`tools/eval.py --cfg ... TEST.MODEL_FILE ...`、`tools/custom.py --cfg ... --p ...`（也可通过 `TEST.MODEL_FILE` 指定）。直接训练须显式指定本次计划所在的输出目录和日志目录。空洞模式、位置、率及 DenseASPP 率的结构错配会被拒绝；边界监督不写入模型张量，手工搬运权重时需核对其原始 run。实测范围见 [迁移验证报告](../test/20261112-pidnet-l-ablation/report.md)。
