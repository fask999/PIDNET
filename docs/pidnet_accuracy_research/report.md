# PIDNet 精度改进研究：先做可复现基线，再查边界与小目标

研究日期：2026-11-10。仓库基线：`4c158cf24ce432f0a8cb43364fae38d93cee0dc3`。用户确认以 Cityscapes、CamVid 为主，精度优先，可以接受适度变慢。本报告给出下一轮实现和消融的顺序；这一轮只做源码、数据清单、权重和文献核验，没有训练，也没有改运行代码。

**建议的第一步：恢复有标签数据，修正权重加载和评估入口，做出一份 train-only → 独立 val 的单尺度基线。** 有了这份基线，再分两个方向做诊断：用同一份权重改推理分辨率，看小目标是否回来；用同一训练配置只改 D 分支的边界监督，看边缘是否更准。如果主要错误是整物体漏检，做定向裁剪；如果主要是类别混淆，先做训练期蒸馏；如果边界确实受限于 1/8 网格，最值得做成论文实验的是 D 引导的局部高分辨率细化，并且要和只看语义不确定性的细化在相同采样预算下正面对比。

配套材料：[源码与复现](repo-audit.md)、[一手文献核对](literature.md)、[实验矩阵](experiment-plan.md)。文中的数字分三类：**本地实测**（代码和模型行为）、**文献报告**（只解释原协议）、**待验假设**（还没有本地精度增益）。

## 读前须知

- **mIoU**：每一类算预测和真实区域的交并比，再对所有类平均。涨 1 个点就是 mIoU 增加 1 个百分点。
- **OHEM**：只挑损失最大的那部分像素参与训练，避免大片容易的背景像素主导梯度。
- **BAS**：本仓库里的边界感知监督。只取 D 分支预测概率大于 0.8 的像素，再算一次主语义损失。
- **KD（知识蒸馏）**：用一个强教师模型在训练时指导学生，推理时教师去掉，学生结构不变。
- **TTA**：测试时增强，比如翻转、多尺度，把多次预测融合。它增加推理开销，和训练期改动要分开记账。

## 1. 本仓库到底在训练什么

共享 stem 把输入降到 1/4，`layer2` 降到 1/8 后分出 P、I、D 三路。P 保持在 1/8，通过两次 PagFM 从 I 选择信息；I 继续下采样到 1/64，用金字塔池化补全上下文；D 接收 I 的两级投影，最终用多通道边界特征控制 P/I 融合。S/M 是 PAPPM + Light_Bag，L 是 DAPPM + Bag。S→M→L 既改容量也改计算路径，所以不能把它当成单模块消融。模型结构见 [models/pidnet.py:17-215](../../models/pidnet.py#L17)，PagFM 和融合见 [models/model_utils.py:268-377](../../models/model_utils.py#L268)。

训练时模型返回 `[P 辅助语义, 融合主语义, D 单通道边界]`，三者都在 1/8 分辨率。部署只取融合主头。默认损失由四部分组成：0.4×P 辅助 CE、主头 OHEM、20×边界加权 BCE、边界区域的主头 OHEM。最后一项只保留 `sigmoid(D 辅助 logits) > 0.8` 且语义非 void 的像素；这个硬选择不向门控传梯度。边缘标签从 trainId 图做 Canny 和膨胀，然后才和语义标签一起缩放裁剪。D 的训练辅助头接在 `layer4_d` 后面，而 Bag 用的是 `layer5_d` 的末级特征，两者不是同一个东西；评估时要分开看 D 的边缘质量和最终语义边界质量。[utils/utils.py:37-60](../../utils/utils.py#L37)、[utils/criterion.py:46-140](../../utils/criterion.py#L46)、[datasets/base_dataset.py:124-187](../../datasets/base_dataset.py#L124)。

PIDNet 作者论文 v3 的 §3、§4.2–4.3 和表 2–4 提供了三条直接依据：

1. L 在 ImageNet 初始化下，从 `Add + Add` 的 80.0 到 `Pag + Bag` 的 80.9 mIoU。这说明两处模块要配合使用，不能把 0.9 个点单独记给 PagFM 或 Bag。
2. S 保持 Light-Bag 时，DAPPM 和 PAPPM 的 mIoU 都是 78.8，但 PAPPM 在 RTX 3090 上的原文测速是 93.2 FPS，DAPPM 是 83.7 FPS。不能默认「换更重的池化就能涨精度」。
3. L 的损失累加是 78.6（主头）、78.8（加 P 辅助）、79.9（加 D BCE）、80.5（加 BAS）、80.9（加 OHEM），都是作者的 Cityscapes 消融。这支持优先诊断边界监督质量，但**不代表**从已经有四项损失的代码出发还能再拿同样收益。原文 §5 也指出，边界标注的准确性会影响这种融合。

## 2. 可用基线与必须先解决的问题

| 来源 | Cityscapes mIoU：S / M / L | 速度与协议 |
| --- | --- | --- |
| [PIDNet 作者 v3 表 6](https://arxiv.org/html/2206.02066v3#S4.SS4) | val 78.8 / 80.1 / 80.9；test 78.6 / 80.1 / 80.6 | 93.2 / 39.8 / 31.1 FPS；RTX 3090、2048×1024、batch 1、Conv-BN 融合；ImageNet 初始化、只做 fine、正式 test 用 train+val；单尺度、不翻转。 |
| [仓库 README:62-75](../../README.md#L62) | val 78.8 / **79.9** / 80.9；test 78.6 / **79.8** / 80.6 | 93.2 / **42.2** / 31.1 FPS。M 的三项和作者 v3 表不同，作为不同来源保留，不能拼表，也不推断差异原因。 |
| [MMSeg PIDNet 模型表](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/pidnet/README.md)、[数据协议](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/_base_/datasets/cityscapes.py#L41) | val 单尺度 78.74 / 80.22 / 80.89；多尺度+翻转 80.87 / 82.05 / 82.37 | 独立实现，用 Cityscapes train、1024×1024 crop、ImageNet 初始化、120k 迭代；模型表测速是单 A100（80.82 / 71.98 / 60.06 FPS）。训练配方有 `cat_max_ratio`、颜色增强、SyncBN，和本仓库不是同一套；多尺度+翻转的分数不能配单尺度速度。 |

[PIDNet v3 表 5](https://arxiv.org/pdf/2206.02066v3) 的 CamVid test 是 S 80.1 mIoU / 153.7 FPS、S-Wider 82.0 / 85.6（README 把后者列为 M）。这组成绩用 Cityscapes 分割权重初始化，CamVid train+val、11 类、960×720、同一块 RTX 3090。它不能和直接从 ImageNet 初始化、或不同 CamVid 划分的实验比较。

原文的 Cityscapes 配方和仓库大致相同：SGD、poly、484 轮、batch 12、1024 crop。但源码的随机缩放包含 2.1 档，论文上限是 2.0。CamVid 源码的学习率是 `.005`，训练截在 121 轮；论文写的是 `.001`、200 轮调度，并在学习率小于 `.0005` 时停止。复现时必须写明选哪一套，不要把它们混成一条基线。[Cityscapes 配置](../../configs/cityscapes/pidnet_small_cityscapes.yaml#L27)、[CamVid 配置](../../configs/camvid/pidnet_small_camvid.yaml#L27)、[tools/train.py:177-181](../../tools/train.py#L177)。

### 本地数据与权重实测

Cityscapes 的 train/val/test 名单是 2,975 / 500 / 1,525 条，CamVid 是 367 / 101 / 233 条。各自 train、val、test 的图像路径不交叉，`trainval = train ∪ val`。

当前 `leftImg8bit` 和 `gtFine` 的软链目标不可达；CamVid 的 `images/`、`labels/` 只有说明文件。逐项检查名单后，存在的图像和标签路径都是 **0**，`output/`、`log/` 不存在，也没有可供对照的本地训练日志。Cityscapes test 名单本来就只有图像路径，正确导出 labelId PNG 后由官方服务器评分；CamVid test 在数据齐备时可以本地评估。[Cityscapes 官方协议](https://www.cityscapes-dataset.com/benchmarks/)、[tools/eval.py:80-111](../../tools/eval.py#L80)。

本地唯一权重 `PIDNet_L_Cityscapes_from_mmseg.pth.tar` 可以用现有 PyTorch 1.13 的 `weights_only=True` 在 CPU 读到 543 个 `model.*` 张量。L 的部署结构匹配 **517/519**，带辅助头的结构匹配 **543/545**；19 类分类头匹配。两个结构都缺 `conv1.0.bias` 和 `conv1.3.bias`，共 128 个元素。仓库 stem 定义了这两个 bias；[MMSeg stem](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/backbones/pidnet.py#L310) 的卷积后面跟 BN，[MMCV `ConvModule` 默认](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/cnn/bricks/conv_module.py#L134) 在这种情况下不创建 bias。旧版加载器把初始化字典合并进去，所以没有报警。

对仓库同一张 2048×1024 样例，用相同预处理和权重，只把构造种子从 2026 改成 2027，主头 argmax 有 **33,967 / 2,097,152 = 1.62%** 的像素不同。只在内存里给这两个缺项补零后，纯推理和带辅助头模式的融合输出完全相同，形状都是 `(1,19,128,256)`。这既说明缺项会破坏复现，也给出了一个无 bias 兼容的候选做法；但**没有**原始权重身份、训练 split 或带标注精度的证据。零填充只能作为明确的格式适配，并和可信来源核对，不能用来掩盖任意缺权重。[本地模型 stem](../../models/pidnet.py#L39)、[eval 加载](../../tools/eval.py#L64)、[custom 加载](../../tools/custom.py#L73)。

### 先处理的项目

| 项目 | 现状与影响 | 最小验收 |
| --- | --- | --- |
| **数据与权重门禁** | 当前数据不可读；`eval` 构造模型时不加载 ImageNet 权重，而是直接要求 `TEST.MODEL_FILE` 指向完整 checkpoint；该文件为空时会去找输出目录的 `best.pt`，本地没有。旧版加载器会把键裁掉 6 个字符，前缀不合时静默少载；当前 `utils/checkpoint.py` 已改为严格匹配，缺键、形状不符、前缀异常都会报错。 | 恢复数据后逐对核对路径、尺寸、原始 labelId/色表和 void。单独构造评估网络；按已知前缀/格式解析，检查主头和各分支的键、shape、缺项白名单与 checkpoint SHA；测试裸键、`model.`、`module.`、错类数，以及两个 stem bias 的明确适配。 |
| **主头和切分** | 8 份 YAML 显式用索引 1；裸默认索引 -1 会选到单通道 D。`MODEL.NUM_OUTPUTS` 不是模型实际输出头的开关。Cityscapes trainval 与 CamVid 自带配方都用样本内 val 选权重，覆盖全部 500/101 张 val。 | 评估时断言最终输出是 `[B,K,H,W]`，只跑同一个主头。开发阶段固定 train→val；发布时先锁定方案，才用 trainval，独立 test 只测一次。记录 split 和权重训练来源；不能先用 test/trainval 教师蒸馏，再宣称 val 独立。 |
| **训练损失与选优** | CPU 已复现空 BAS/OHEM 的 `IndexError`、同分空均值的 NaN、关闭 OHEM 时 SB 形状错误。训练像素 Acc 把 void=255 算进分母（独立 mIoU 仍会忽略）；恢复 checkpoint 时 best 元数据先写后更新，可能覆盖更好的模型。 | 有效像素为空时返回与图相连的有限零；在非空集合上验证阈值和「至少保留 K 个」的约束；普通 CE 显式支持张量/头列表。用全 ignore、空边缘、单像素、阈值并列和带反传的常规样本验证；用 `.70→.80→恢复→.75` 验证 best 不倒退。把兼容修复和真实训练改法分开记账。 |
| **尺寸与增强** | 两个测试增强字段目前没人读取。`TEST.IMAGE_SIZE` 传进了数据集，但原单尺度不 resize；非 8 倍数输入的 D 加法可能失败；Cityscapes test 非原尺寸时，备用路径把 `W,C` 当成 `H,W`。 | 显式 pad 到 8 的倍数，还原尺度并裁掉 padding；测奇数 H/W；`scale=1, flip=false` 要和修复后的单尺度主头一致，逆翻转坐标检查通过。固定输出 labelId。 |

入口证据：[tools/train.py:92-125](../../tools/train.py#L92)、[恢复和保存 :166-205](../../tools/train.py#L166)、[configs/default.py:79-97](../../configs/default.py#L79)、[utils/function.py:131-202](../../utils/function.py#L131)、[models/pidnet.py:169-215](../../models/pidnet.py#L169)。[完整源码审计](repo-audit.md#问题影响及最小验证) 区分了触发条件、指标影响和设计取舍。边缘含 void、随机裁剪分布、1/8 网格容量是下一步的**待量化瓶颈**，不能只凭源码断定它们是真实数据上最大的误差来源。

## 3. 精度改进优先级

编号与[实验矩阵](experiment-plan.md)一致。先过第 2 节的基线门禁。编号按首轮投入和当前证据排序；R5 是较值得做的论文方向，但应该由误差分析触发。训练期改动只要不改变导出的学生图，理论上不增加推理算子，但仍要在同一设备上复测。

| 编号 | 做什么、和谁对照 | 看什么、有什么风险 |
| --- | --- | --- |
| **R1：同权重扫输入分辨率，另测 flip/TTA** | 在 [BaseDataset.inference:190](../../datasets/base_dataset.py#L190) 和 [testval:131](../../utils/function.py#L131) 明确提取主头 logits。固定权重和标注尺寸，测原图 1×、0.75×、1.25× 单尺度；另测 1×+flip 和少量尺度的离线融合。缩放、按需翻转后 pad，输出恢复至该尺度的 padded 图，先裁掉 pad 再逆翻转并回到原图坐标；在原图上取 softmax 后等权平均，别直接平均 `.exp()` 的分数。 | 不需要重训，最先判断当前失分是否随分辨率改善。若 1.25× 改善小类，再测更接近线上开销的 1.125× 单尺度；它的理论像素数约为 1.27×，但不等于实测时延。flip 要两次前向，多尺度更贵。只有「单尺度稍放大」满足延迟门槛才进入线上候选，重 TTA 作为离线上限。MMSeg 同实现同权重的 SS/MS+flip 差值只能说明有必要测，不能许诺本仓库的涨幅。 |
| **R2：改 D 的边界监督与 BAS 采样** | [gen_sample:154](../../datasets/base_dataset.py#L154) 保留原 Canny 作对照：先只在旧边缘目标中排除 void/pad 邻域；再于几何变换后，从有效异类邻接生成固定像素宽的目标，并显式把有效掩膜传给 [weighted_bce:107](../../utils/criterion.py#L107)；[BAS:52](../../utils/utils.py#L52) 先做开/关子消融；只有证据指向硬阈值后，才另试「GT 边界带向预测门过渡」或软权重。 | 看 void 邻域占比、正负平衡、D 边缘分数、最终语义边界 F1/窄带 IoU 和整体 mIoU。可能产生粗边、伤细物体，或边界为空；主 loss 和网络保留。PIDNet 表 4 直接支持边界训练的重要性，Gated-SCNN、ABL 是间接证据，不能把涨幅相加。 |
| **R3：定向裁剪稀有类和小实例** | [rand_crop:75](../../datasets/base_dataset.py#L75) 目前均匀取位置，OHEM 只能在已经见到的像素里挑难例。先统计每类有效 crop 覆盖、实例被完整/部分保留的比例；如果确实缺失，只把一部分 crop 换成关键类/小实例中心，其余保持随机，更新步数不变。颜色增强、最大类占比约束另测。 | 目标是小类 IoU、实例召回/iIoU 提升，同时检查邻近背景误检和整体 mIoU。CamVid 连续帧应按视频/片段做误差归因，不能只盯重复画面。文献支持类别均衡思路，但没有 PIDNet 单因素数据。 |
| **R4：强教师做训练期蒸馏** | 在 [FullModel.forward:37](../../utils/utils.py#L37) 只对融合 logits 加 KL。先从有可核训练来源、同 split、同类序的 L 或 SegFormer 教师开始，比较不蒸馏、逐像素 KL、CWD 空间归一化 KL；同温度、同训练步数、同掩膜。CamVid 需要 11 类教师，或明确做 19→11 的任务转换验证，不能直接按通道蒸馏。 | 适合语义混淆和无额外在线成本的路线。CWD 在 PSPNet 上有明确对照，但 PIDNet 已经有 I 上下文；在双分支的 DDRNet 上，SCTNet 的 SIAM 只报 79.5→79.6（Cityscapes val 全分辨率，表 5），不要套用单分支的提升。验证教师在关键类和窄边带上确实更强，并记录离线存储、对齐和训练代价。 |
| **R5：用 D 和语义不确定性选点，补 1/4 细节** | 在 [PIDNet.forward:169-215](../../models/pidnet.py#L169) 保留 `layer1` 的 1/4 浅层特征，基线仍是 1/8 融合主头。首轮显式接入已有的 D 边缘头，计入耗时；如果超时，再单测更轻的 1×1 读出。先做等预算的随机点、top-two margin、D-only、D+margin；只在被选位置取浅层特征和粗 logits，用小型残差头回写，初始残差置零。另做同类浅层稠密轻头作为控制。 | 原文有 1/8 上的 D 引导融合，PointRend 也验证了不确定点细化；贡献要证明 **D 额外提供的信息**，并报告相同采样预算和 p95 延迟下的边缘/小实例收益。D 可能漏掉被粗图完全吞掉的对象；采样点稀疏也可能在小 GPU 上因为 gather/scatter 变慢。 |
| **R6：定向增强 I 上下文** | 先确认错误在物体内部、长距离类混淆，还是大区域不一致。只改 [PAPPM/DAPPM:114](../../models/model_utils.py#L114) 中一个低分辨率上下文路径：先改一个池化感受野，保持网格、主干、通道和训练损失不变；再独立对照轻量尺度门控，参数量保持相近。S/M/L 在同预算下的独立容量对照另列。 | 若收益主要在物体内部，且同设备延迟可接受，再试轻量 attention 或中层分支。[PP-LiteSeg 表 3](https://arxiv.org/pdf/2204.02681)、BFMSNet 支持多尺度语义聚合的思路；PIDNet 原表 3 不支持直接换重池化提分。 |
| **R7：主头区域损失或边界对齐损失** | 保留[原主头 OHEM](../../utils/criterion.py#L46)，先在全体有效像素上加一个 Lovász 项，不用 OHEM 子集近似整幅 IoU；若确认主要是已检出物体的边界错位，再单做 ABL，对比同一 Lovász 基线，并保留原方法的梯度截断。 | ENet 的 Lovász 原表 4 里 IoU 上升、iIoU 反而下降；加入后必须看小实例和逐类结果。此项排在有误差诊断之后，避免同时调多个 loss 权重。 |
| **R8：预训练和额外训练数据** | 先统一可追溯的 ImageNet 初始化；CamVid 比较同一配方下「从相应 Cityscapes 分割 checkpoint 迁移」和「既定起点」。额外数据另外开组：训练分区去重、冻结强教师伪标签，对比 fine-only、额外伪标签、独立强初始化；低置信伪标签不作为 D 的硬边界。`DATASET.EXTRA_TRAIN_SET` 现在没接线，需要新数据路径和独立记录。[配置 :50](../../configs/default.py#L50)、[训练数据入口 :92](../../tools/train.py#L92)。 | 训练成本、域差异、外部数据和协议变化最大。UniMatch V2 全 fine 加约 20K 无标签的同研究对照是 85.2→85.5，不能套用到少标注条件；DINO 系列作为教师需要已经训练好的语义解码器，不能直接加载到 CNN。把额外数据组和纯 fine 组分表。 |

### 哪些近年结果值得参考

- [PointRend，CVPR 2020，表 6–7 及补充表 1](https://arxiv.org/html/1912.08193) 是 R5 的直接方法先例。Cityscapes 语义 val 同 DeepLabV3-OS16，mIoU 77.2→78.4；[同组 8 px 窄边带](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Kirillov_PointRend_Image_Segmentation_CVPR_2020_supplemental.pdf) 42.4→47.3，无 TTA。这是 DeepLabV3 上的细化证据，不能外推到「PIDNet 已经使用 1/8 高分辨率 P/D」的情况；原表也没给 GTX 1650 速度。另有 [Marin 等 ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Marin_Efficient_Segmentation_Learning_Downsampling_Near_Semantic_Boundaries_ICCV_2019_paper.pdf) 已经用语义边界导引采样，论文贡献不宜写成「首次用边界做稀疏处理」。
- [SCTNet，AAAI 2024，表 5–6](https://arxiv.org/html/2312.17071#S4) 对同 SCTNet-B 全尺寸 Cityscapes val，训练期语义对齐 78.5→80.5；同表 DDRNet-23 是 79.5→79.6。SCTNet 附录列了 ImageNet-1K 初始化，该表没有单列 DDRNet 的初始化。教师只用于训练，所以值得用同 split 做 R4；学生原本就有强语义分支时，收益可能较小。该表没有单列 TTA。结合[逐像素/结构蒸馏](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) 和 [CWD 表 4](https://arxiv.org/pdf/2011.13256)，先做可解释的目标对照，不要一上来就叠复杂对抗损失或全像素关系损失。
- [GCNet，CVPR 2025，§3.2、表 2–3](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Golden_Cudgel_Network_for_Real-Time_Semantic_Segmentation_CVPR_2025_paper.pdf) 说明可以把训练时的多路径卷积重参数化进部署卷积。它在 Cityscapes val 的同表结果是 1024×2048、A100、Conv-BN 融合、无 ImageNet：重训 PIDNet-S 76.4 / 128.7 FPS，GCNet-S 77.3 / 193.3 FPS。该表没有单列 TTA，也不能和作者 ImageNet 初始化的 PIDNet-S 78.8 / 93.2 拼成一条对比。重参数化可以留给将来受延迟约束的备选，但要验证转换前后的输出；当前优先查边界。
- [BFMSNet，IET Image Processing 2026，§4.2/4.4，表 1/2/6/8](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70287) 在 Cityscapes val、同 ResNet18/ImageNet 底模下，77.22→77.71（加边缘融合），再加混合边界损失到 78.21，再加上下文到 78.53；§4.5 的 test 是 78.2，不能把 val 数抄成 test。它验证了不同结构里的趋势；比较表里有混合分辨率，不能据此推断 PIDNet 换模块的绝对收益或测速，只能作为次一级证据。
- [OpenMMLab 的配置](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py#L66) 给出了易复现的 `cat_max_ratio` 和 photometric 增强参照，但模型表不提供这两项的独立消融。[UniMatch V2 表 XIII](https://arxiv.org/html/2410.10777v2)、[DINOv3 表 24](https://arxiv.org/html/2508.10104v1#S6.SS3) 的设置包含额外图像或大规模预训练，适合证明教师/数据路线值得做对照。其余边界、损失、教师和 TTA 的原表及限制见[文献附录](literature.md)。

### 论文验证与工程提分各怎么走

**工程路线。** 修好加载、split、输出和 loss 后，用 R1 的同权重诊断和 R2 的边界目标作为第一轮。如果要求接近原延迟，优先留下 R2/R3/R4 中有可复现收益的训练期改动。如果允许小幅时延增加，实测 1.125× 单尺度和 R5 的轻量局部头；M/L、重 TTA、外置 SegFix 分别列为容量或离线上限。取实测 mIoU、关键类和 p95 的 Pareto 前沿；`p95 ≤ 基线 1.25×` 可以作为第一轮筛查门槛，但用户并没有指定这个硬指标。

**论文路线。** 用 R1 的分辨率诊断和边缘/内部错误分解，先验证细目标在 1/8 粗图上确实无法分开。提出可证伪的假设：**在同等采样预算下，D 边界信号和语义 margin 联合选点，比只看 margin 更能找回薄结构，并让更多局部计算落在真正需要的区域。** 比较 B0、随机、D-only、margin-only、联合、稠密轻头；点数和浅层通道匹配，做边界质量对采样预算及 p95 的曲线。用真值边界带选点只能估计上限，不能参与部署成绩；用「到边界的距离分桶」和细物体召回验证机制，而不是只给总体 mIoU。如果联合门控在相近延迟下超不过 margin-only 或稠密轻头，就停止扩展结构，改做边界目标和采样失效的实证分析。

## 4. 评估和实际验证

把 Cityscapes train（2,975）→ val（500）作为主筛查，CamVid train（367）→ val（101）作为迁移验证。CamVid 是小样本，报告配对种子和最好结果时要加片段分组误差。发布时才把已经冻结的配方用于 trainval：Cityscapes 提交 test PNG，CamVid 测 233 张独立 test。先用 S 做工程筛查；算力允许时给 M 单独跑一套精度基线，L 作为容量对照。

B0 要锁定 commit、权重哈希、数据名单哈希、类映射、同一输入/输出尺寸、插值方式、优化器步数、BN 策略和硬件。用同种子快速筛选，入围方案再用至少三组配对种子（建议 304、305、306），报告每个种子的差值和均值/离散程度。小预算训练只能筛方向，不能替代最终训练结果。

主指标是原图、有效像素上的 19/11 类 mIoU 和逐类 IoU。Cityscapes 特别看 pole、traffic light、traffic sign、person、rider、bicycle、motorcycle；CamVid 按实际的 11 类命名报告细类和小目标。固定语义边界提取、void 带处理和 3/5 px 匹配容差，报告最终语义边界 F1 或固定宽度 trimap；D 头的检测效果另列，不混进主指标。Cityscapes 实例标注齐备时用[官方 iIoU](https://www.cityscapes-dataset.com/benchmarks/) 和固定真值面积桶（同时报样本数及匹配定义），非实例类单列。测速用同硬件、同输入、同精度、batch=1，预热并与设备同步，分别记录模型前向和端到端的 p50/p95、峰值显存；CPU 烟测、原作者 RTX 3090、MMSeg A100、部署设备的数字都单列。[具体矩阵与门槛](experiment-plan.md)。

本机的 `segformer` 环境是 PyTorch 1.13.0 / NumPy 1.21.6 / OpenCV 4.1.2，CUDA 可见 GTX 1650 4 GiB。小 CPU 张量、路径检查和真实单帧前向可以立即做；数据恢复后先验证单图标注流程。默认两卡 batch 12 的 1024 crop 长训、以及重教师并行，应该按可用的外部 GPU 和显存来安排。如果只用本机做探索，先试小 batch/crop 或训练期离线教师缓存；梯度累积不等价于大 batch 的 BatchNorm，PPM 的全局 1×1 BN 在每副本 batch=1 时也可能报错。改变这些条件要重建相同设置的基线，不能沿用论文分数。[训练 DataParallel](../../tools/train.py#L145)、[PPM 全局池化 BN](../../models/model_utils.py#L133)。

实际执行过的数据/权重命令如下。脚本只在 CPU 做前向，原始 checkpoint 不落盘重写；日志含文件 SHA、缺键、尺寸和比较计数，没有测 FPS 或 mIoU。另外实际核对了 8 份 YAML 的合并值、模型输出、错误尺寸，以及空 OHEM/关闭 OHEM 的合成行为，见[源码审计的复现](repo-audit.md#复现与未决)。

```bash
cd /home/robot/catkin_ws/PIDNET
/home/robot/anaconda3/envs/segformer/bin/python -B \
  .super-agent/evidence-fe564f13e683481fa021a45363a7bba7/local-baseline.py
```

证据：[本地检查脚本](../../.super-agent/evidence-fe564f13e683481fa021a45363a7bba7/local-baseline.py)、[实际输出](../../.super-agent/evidence-fe564f13e683481fa021a45363a7bba7/local-baseline.log)。待补证：恢复并核对标签、确认可用于独立验证的权重来源，再跑 B0 和首轮消融。当前不能判断哪一项能给本地数据带来多少精度提升。

---

> 修订记录：2026-11-11，ZCode（opencode-go-chat/deepseek-v4.1-flash）重写全文，调整结构与表述，补充术语说明，并核对源码行号与链接。保留原始结论与数据来源，按当前代码修正了少量表述。
