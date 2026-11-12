# PIDNet 精度改进：一手文献与迁移边界

这里的数值是原文报告，未在本仓库复现。只在同一研究的同协议对照内解释差值。`val` 是验证集，`test` 是测试集；SS/MS 是推理时单/多尺度；TTA 是测试时增强。预训练来源、额外分割训练数据、验证集是否进入训练以及滑窗/翻转应分别记录。文中“未报告”表示引用的对应实验没有单独说明该字段。所有分数按百分数表示，差值以百分点计。

## 边界与小目标

### 1. Boundary IoU 和 Cityscapes iIoU：先找出失分在哪里

- **原文证据：** [Cheng 等，CVPR 2021，§3–5、表 1](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf) 分别取预测与真值掩膜各自的内侧边界带，再做 IoU。它对大物体边界偏移比区域 IoU 更敏感。论文与[作者 API](https://github.com/bowenc0221/boundary-iou-api)在 Cityscapes 支持实例/全景任务，Boundary AP/PQ 不能直接当作 19 类语义 mIoU。Cityscapes [官方指标](https://www.cityscapes-dataset.com/benchmarks/)另有 class iIoU：按“该类平均实例面积 ÷ 当前真值实例面积”加权 TP/FN，FP 保持像素计数。只覆盖可评实例类别；pole、traffic light、traffic sign 不能靠这个指标诊断。
- **最小迁移：** 固定 Cityscapes val 的同一批原分辨率预测，报告 mIoU、19 类 IoU、排除 void 的语义边界 F1（例如 3/5 px，固定边界提取、匹配容差与聚合方式），以及有实例标注时的官方 class iIoU。调用[官方评估器](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py)前，把 train ID 转回 label ID，并使用 `gtFine_instanceIds`。要定位小目标漏检，再按真值实例面积分桶报召回和样本数；细杆/信号灯/标志仍看逐类 IoU 与边界 F1。若实现逐语义类 Boundary IoU，应注明这是本地扩展，并固定带宽和 void 处理。边界 F1 应从**最终语义输出**计算，D 分支的二值边缘分数另列。

### 2. Gated-SCNN（ICCV 2019）：先改已有 D 监督，再考虑加门控

- **原文证据：** [Takikawa 等，§3.1–3.2、§4、表 3–4](https://arxiv.org/pdf/1907.05740) 用高层语义门控浅层形状流，联训区域和边界。Cityscapes **val、fine-only**：表 3 的 ResNet-101 基线加 GCL，mIoU 72.7→74.3，5 px 边界 F1 69.8→73.3；WideResNet 组 mIoU 79.2→79.8、F1 73.0→75.9，进一步加图像 Canny 梯度为 80.1/75.6。表 4 的 ResNet-101，GCL 后加 dual-task 使 3 px F1 65.0→68.0。上述 val 消融未分别说明预训练来源或 TTA；§4 明说 test 最佳模型用 0.5/1/2 倍，这项不能回填到 val。表 3 显示边界分数和整体 IoU 可以朝不同方向变化。
- **最小迁移：** PIDNet 已有高分辨率细节 P、语义 I 与边缘 D，不必复制整条 shape stream。先在 [gen_sample](../../datasets/base_dataset.py#L106) 对照现有“在类编号图上 Canny、膨胀后再随图缩放”的做法，测试**缩放/裁剪后**由有效异类邻接生成固定像素宽的边界。void、裁剪填充和无有效边缘的 crop 要明确处理；修改 D 的目标时，对应把有效像素掩膜传给 [边缘 BCE](../../utils/criterion.py#L102)，否则把 void 置零等同于新增负样本。只在错位/漏检证据指向融合时，再试在一处 D 或 Bag/LightBag 的门控调整，单列延迟。

### 3. SegFix（ECCV 2020）：验证“边界可修、内部已对”这一假设

- **原文证据：** [Yuan 等，§3、§4、表 2/4/6](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf) 训练独立的边界掩膜和朝内部的离散方向，用内部像素类别修正边界。Cityscapes **val**，DeepLabv3 + Higher-HRNet 修正，表 4 mIoU 79.5→80.5，边界 F1 56.6→60.9；该表未单列底模骨干、预训练/额外数据或 TTA。原文消融主要用 1 px F1 容差，不能与 Gated-SCNN 的 3/5 px 分数直接比较。[补充材料表 2](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477-supp.pdf) 中 pole、traffic light、traffic sign 的 IoU 分别 +3.1/+2.7/+2.4；这是同一 val 设置的逐类值。表 6 在 **MS Gated-SCNN** 底图上仍见 mIoU 81.0→81.5、F1 61.4→63.1；修正器 TTA 未单列。表 1 使用真值边界和方向做上限探测。表 2 的 HRNet-W18 修正器自身为 **16 ms**（2048×1024、单张 V100、PyTorch 1.4 + TensorRT）；Higher-HRNet 为 69 ms，这些都不含 PIDNet。
- **最小迁移：** 先在固定 val 预测做诊断：仅对真值边界带做偏移上限实验，观察哪些错误是错位，哪些是整物体漏检；明确标为使用真值的上限。部署候选才在 D 分支附加小型方向头，或者单列外置 SegFix 做精度上限。检查细物体无可靠内部像素、目标邻接、void 穿越和坏方向时是否需要禁用修正。D 的二值边界图本身不能替代方向预测。

### 4. Active Boundary Loss（AAAI 2022）：有边界但位置不准时的训练选项

- **原文证据：** [Wang 等，§3–4、表 1/4](https://ojs.aaai.org/index.php/AAAI/article/download/20139/19898) 从语义预测找边界，再监督它朝最近真值边界的方向移动。Cityscapes **val、fine（无 coarse）、SS**，DeepLabV3/ResNet-50 的 CE、CE+Lovász、CE+Lovász+ABL 分别 79.5、80.2、80.5 mIoU；不做指定梯度截断降为 76.0。预训练和 SS 是否翻转未单列。表 4 的 OCR/HRNetV2-W48 82.2→82.9 是 MS 结果，不能当成表 1 的追加提升。
- **最小迁移：** 给 Bag/LightBag 的最终 logits 单独加训练损失；保留 D 的 BCE 和现有语义边界损失，先用同一 CE/OHEM+Lovász 对照，逐项测 ABL。依原文保留梯度截断，监测有效边界数、空样本和 AMP 下的数值。该方法依赖预测中已有边界，对整目标漏检无直接修复；PIDNet 已监督边界，收益未证实。

## 训练目标与裁剪

### 5. Lovász-Softmax（CVPR 2018）：优化区域 IoU 后必须复查小实例

- **原文证据：** [Berman 等，§4、图 9、表 4](https://arxiv.org/pdf/1705.08790) 从 ENet 作者权重继续训练。Cityscapes **test** 的 class IoU 58.29→63.06，class iIoU 却从 34.36→34.06；[原 ENet 提交](https://www.cityscapes-dataset.com/method-details/?submissionID=132)和 [Lovász 提交](https://www.cityscapes-dataset.com/method-details/?submissionID=993)均标 fine、未列外部数据，TTA 未报告。图 9 对照加权 CE 继续训练及不同 batch size；更大 batch 在这一 ENet 配方中更有利。ENet 的旧基线差值不能当 PIDNet 预测。
- **最小迁移：** 在现有主语义 OHEM 旁，仅对最终语义 head 的有效像素加一个训练期 Lovász 项；保留辅助 CE 与边界项。不直接拿 OHEM 筛后的像素估整个 batch 的 IoU，注明按图、按卡或按全局 batch 聚合。用相同微调步数、初始权重、裁剪、评估方式对照。若 mIoU 提高但 iIoU 或关键小类下滑，先按类定位变化再决定是否采用。

**裁剪采样是另一条变量。** Cityscapes S 的[配置](../../configs/cityscapes/pidnet_small_cityscapes.yaml#L21)已用 OHEM（阈值 0.9、`min_kept=131072`），[损失实现](../../utils/criterion.py#L63)在主 head 与预测边缘选中的语义像素中挑难样本；它无法补回训练裁剪中完全没出现的类。[当前裁剪](../../datasets/base_dataset.py#L60)均匀取位置。先统计 crop 内关键类出现率；若缺失，再只改一部分 crop 围绕罕见类或小实例取样，保留随机 crop 作对照，记录类/实例覆盖和背景误检。Gated-SCNN §4 的 test 配方和 [NVIDIA 的类别均衡采样实现](https://github.com/NVIDIA/semantic-segmentation/blob/main/datasets/uniform.py)可供实现参照；所引最终成绩没有把采样单独消融。加边界掩膜或稀有类裁剪时应覆盖全 ignore、空边界与无 OHEM 入选像素：[当前 OHEM](../../utils/criterion.py#L63)对空数组的索引/均值需要单独防护。

## 蒸馏与教师来源

### 6. Structured Knowledge Distillation（CVPR 2019）：从逐像素 KL 开始

- **原文证据：** [Liu 等，§3、§4、§5.1–5.2，表 1](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf)，[会议版代码](https://github.com/irfanICMLL/structure_knowledge_distillation/tree/cvpr_19)。Cityscapes **val、fine-only**，PSPNet/ResNet-101 教师 78.56 mIoU；ImageNet 初始化的 ResNet-18 学生三次均值：69.10（无蒸馏）、70.51（逐像素 PI）、71.37（PI+像素关系 PA）、72.67（PI+PA+全图对抗 HO）。对应表未单列 TTA。与随机初始化学生相比，ImageNet 初始化基线更强，应据表中的对应行比较。
- **最小迁移：** 先冻住只见过训练集的 PIDNet-L 或 19 类大教师，在 `FullModel` 的[最终语义输出](../../utils/utils.py#L37)上做同一坐标、同一类别顺序的 `KL(teacher || student)`；忽略 void，并与无蒸馏同训练预算比较。温度和权重另作消融，不先叠加 PA/HO。若逐像素 KD 只改善大区域，可用低分辨率或局部关系试 PA；全位置关系矩阵随空间位置数平方增长，对抗训练还引入稳定性变量。导出模型不携带教师。

### 7. Channel-wise Knowledge Distillation（ICCV 2021）：低改动的结构蒸馏对照

- **原文证据：** [Shu 等，§3、§4.1–4.4，表 2/4/5](https://arxiv.org/pdf/2011.13256)。Cityscapes **val，fine，无 coarse，SS**；表 4 以 PSPNet-R101 为教师（78.56），PSPNet-R18 为学生，三次平均 mIoU 69.10→71.74（逐像素 logits KL）或 74.87（通道 logits KL）；蒸馏 feature map 为 74.27。§4.2 只确认单尺度，未另述翻转。表 5 的 70.09→75.90 是另一组 ImageNet 初始化学生结果；不能和表 4 拼接成增益。表 5 也给出不同类型学生 DeepLab-R18 的结果，支持测试跨架构，不证明在 PIDNet 同样有效。
- **最小迁移：** 用同一教师和同一对齐后的 19 类 logits，只把归一化轴从“每个像素上的类别”切换到“每个类通道的空间位置”，teacher detach，显式固定方向和温度。无有效位置要跳过；比较相同的 void mask 与网格，并检查稀有类的空间分布。先不用中间 feature 的通道投影，避免把投影收益和 KL 差异混在一起。

### 8. SegFormer（NeurIPS 2021）：优先验证教师类别覆盖与训练来源

- **原文证据：** [Xie 等，§4.1/4.3、表 1(a)/4](https://proceedings.neurips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)。MiT-B5、ImageNet-1K，Cityscapes **val** mIoU 82.4 SS / 84.0 MS；原文采用 1024×1024 滑窗，[作者 B5 配置](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py)给出 stride 768。**test** 另用 train+val，ImageNet-1K、无其他分割数据为 82.2，加 Mapillary Vistas 预训练为 83.1；表 4 未单列翻转/多尺度。test 权重会污染本地 val 蒸馏协议。
- **最小迁移：** 候选优先找明确 train-only、输出 19 类的 B5 权重，按自己的输入规范独立评估教师，对齐增强与 `align_corners` 后再比较 PIDNet-L / B5。表 1 的 SS/MS 差值是该教师的推理差值，不是学生获益。若离线缓存，必须记录尺寸、裁剪/翻转对应和类别 ID；原图缓存后再裁与同 crop 在线计算的上下文并不等价，应固定一种做对照。

### 9. Mask2Former（CVPR 2022）：可作更强的语义教师，别误取实例模型

- **原文证据：** [Cheng 等，§4.5、表 6](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)及[附录 B.1、表 VII](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Cheng_Masked-Attention_Mask_Transformer_CVPR_2022_supplemental.pdf)。Cityscapes **val**，语义模型 Swin-B（ImageNet-22K 预训练）为 83.3 SS / 84.5 MS mIoU，Swin-L（同预训练）为 83.3/84.3；附录注明这组未用额外分割数据，SS 整图 1024×2048，具体翻转设置在此表未单列。表 7 的 panoptic 训练语义转换成绩和专训 semantic 成绩不同。
- **最小迁移：** 使用 Cityscapes **semantic** 模型，在 query mask 与类别分数组合后的稠密 19 类输出上蒸馏。该输出是组合分数，不应当成与 PIDNet 同定义的原始 logits；先定义逐像素归一化并验证数值，再做 KL。先核其边界/小类质量是否确实强于 PIDNet-L，再决定更重的教师是否值得算力。

### 10. DINOv3（2025 技术报告）：较新教师证据，含外部预训练

- **原文证据：** [Siméoni 等，§3.1、§6.3.2、表 24、附录 D.10](https://arxiv.org/html/2508.10104v1#A4.SS10)。冻结 DINOv3 ViT-7B/16，训练 ViT-Adapter+Mask2Former 解码器；Cityscapes 表 24 为 86.1 SS / 86.7 TTA mIoU，输入基准 1280，SS 也使用滑窗，Cityscapes TTA 的倍率为 1/1.5/2，Cityscapes 翻转与具体评估 split 未在该表/附录单列。骨干用大规模网络图像自监督训练（数据组成见 §3.1，含 LVD-1689M）；语义解码器协议还列 COCO-Stuff、Hypersim 预训练，表 24 未单列各 Cityscapes 阶段的数据消融。冻结骨干仍须训练重解码器；表 24 也没有超过该表所列既有 Cityscapes 对照（86.3/87.0）。
- **最小迁移：** 当已有对应数据和语义解码器时，只把 DINOv3 作为离线教师试同样的 logits 方案；若只有骨干，则需训练城市街景解码器或另作特征蒸馏，两者分别记账。ViT 权重不能直接加载到 PIDNet CNN：本仓库的[加载器](../../models/pidnet.py#L184)仅匹配同键且同 shape。未经同 split 检验的教师结果只说明候选方向。

### 11. UniMatch V2（TPAMI 2025）：完整标注下的增益比低标注设定小

- **原文证据：** [Yang 等，§III-C、§IV-B/D/E、表 II/XIII](https://arxiv.org/html/2410.10777v2)，[作者实现与训练日志入口](https://github.com/LiheYoung/UniMatch-V2)。用 EMA 教师为弱增强图像打伪标签，对学生的强增强与互补 dropout 做一致性训练。Cityscapes **val**，表 II 的 DINOv2-B+DPT（DINOv2 为外部自监督预训练，见 [DINOv2 §3](https://arxiv.org/html/2304.07193#S3)）在 186 张有标签、剩余 fine train 按无标签处理时，labeled-only 80.8→83.6 mIoU。原文 Cityscapes 用 798 滑窗，半监督结果报告 EMA mIoU，未注明多尺度或翻转。表 XIII 的全量约 3K fine 加约 20K Cityscapes Extra 作为无标签池是 85.2→85.5（表 XIII 未另列骨干；85.2 对应表 II 的 DINOv2-B 全监督行）。默认置信阈值 0.95；§IV-D6 的 Cityscapes 部分标注消融中，0 的表现反而最好，不能照搬固定阈值。
- **最小迁移：** 只有拿到合法、与 val/test 去重且来源明确的额外训练图像时再启动。先对照 train-only 教师的冻结离线伪标签与 fine-only；记录阈值下逐类像素接受率和近边界错误，在小类和边界上先审标签质量。给伪标签单独加语义损失/权重，不直接用低置信伪标签生成 D 的硬边界监督。若离线方案有收益，再评估是否值得引入 EMA/强弱增强的在线方案。

## 多尺度与实验顺序

### 12. Hierarchical Multi-Scale Attention（2020）：TTA 是离线对照，也可生成训练目标

- **原文证据：** [Tao 等，§3.2、§5.1.1、表 1–3](https://arxiv.org/pdf/2005.10821)。**Mapillary val**，DeepLabV3+/ResNet-50，1× SS 为 47.7，0.5/1/2× 平均为 49.4，学习式融合为 51.6 mIoU；多加 0.25× 后平均降到 48.7，学习融合到 52.2。该表未单列外部预训练或翻转；表中的 FLOPs 与 V100 minibatch 时间是**训练开销**。**Cityscapes val**，HRNet-OCR 表 2 平均融合 84.9→学习融合 85.4；“未用自动标签”对照用原 coarse 标签，并非 fine-only，val 子实验预训练未单独明示。**Cityscapes test** 85.1 来自 Mapillary 预训练、train+val fine、coarse 图像自动标签、0.5/1/2× 与翻转的组合，不用于单因素或实时比较。
- **最小迁移：** 同一 PIDNet 权重先测固定 SS、SS+flip、少数尺度融合三档，复原原图坐标后在统一的 logits 或概率域聚合（[现有 inference](../../datasets/base_dataset.py#L136)返回 `exp` 后的分数，不能不加说明就当作已归一化概率）。记录输入倍率、裁窗、翻转、整图端到端延迟与峰值显存。多尺度如果主要改善小目标，可用**训练图像**的多尺度高置信结果教单尺度学生；如果只有离线得分上涨，应把它保留在独立的精度档。

建议先锁住模型版本、权重、Cityscapes fine train/val、19 类映射、原图 SS 与基准计时方法。第一轮用同一组预测补齐边界 F1、iIoU 和关键类 IoU，判断是细边错位、实例漏检还是类混淆；随后只变一个因素：细边错位先试 D 边界目标，漏检先试定向 crop，类混淆先试同数据教师的逐像素/CWD 对照。Lovász、ABL、额外训练数据与 TTA 分列记录。对训练候选至少用同种子或配对种子、同训练步数和同评估配置；原文差值不能相加，也不能当成 PIDNet 的预期增益。
