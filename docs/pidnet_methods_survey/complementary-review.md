# 与补充报告的交叉核查

本页对照用户提供的 [PIDNet Accuracy Improvement Report](../PIDNet_Accuracy_Improvement_Report.md) 与 [本次文献主表](report.md)。原报告的优势是把失分现象对应到 PagFM、PPM、D 边界和训练配方；本页补的是原文证据、草案可验证性质和试验顺序。**摘要及 §4.1 的预估 mIoU、毫秒、参数量不作为本仓库结果**。这轮没有训练 PIDNet，也没有做端到端计时。训练时改变目标而保持推理图，是可验证的设计约束；不能承诺 mIoU 或实测毫秒。

## 新增论文证据

| 补充方法 | 核对到的结果 | 对当前 PIDNet 的用法 |
| --- | --- | --- |
| **FreqFusion**，Linwei Chen 等，TPAMI 2024，[作者全文 §IV、表IV](https://arxiv.org/html/2408.12879v1#S5.SS1)、[实现](https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py) | 同论文 Cityscapes val 单尺度，SegFormer-B1 78.5→80.1，243.7→271.8 GFLOPs（1024×2048）；UPerNet-R50 78.8→79.8。作者对照改多尺度融合，非 PIDNet 增益。代码接收高/低分辨率两路，组合自适应低通、高通和偏移；依赖 MMCV CARAFE 或仓库给出的 PyTorch 替代，速度和显存仍需测。 | 如果基线主要是 I→P 对齐问题，先只测一处 PagFM 的 2× 融合，明确对齐投影与 value、融合接口及 odd-size 形状；不要把它当可全局替换的 interpolate。FADC 改卷积，FreqFusion 改融合，二者不是同一算子。 |
| **DySample**，Wenze Liu、Hao Lu、Hongtao Fu、Zhiguo Cao，ICCV 2023，[作者全文 §3、表4](https://arxiv.org/pdf/2308.15085)、[实现](https://github.com/tiny-smart/dysample/blob/main/dysample.py) | ADE20K SegFormer-B1，双线性 41.68 mIoU / 27.80 bIoU，DySample-S+ 43.58 / 29.93；不是 Cityscapes 或 PIDNet。作者单算子 256×120×120 的 ×2 延迟示例为 1.6→6.2 ms，不能移植该数字。代码用整数尺度、通道分组和 grid_sample，无额外 CUDA 扩展。 | 优先作为一处 I→P 的 2× 采样对照。先验证尺寸、边界、梯度和导出后端；未证明能等价替换多处不同尺度或最终 logits 插值。 |
| **Strip Pooling**，Hou、Zhang、Cheng、Feng，CVPR 2020，[原文 §3、表2–3](https://arxiv.org/pdf/2003.13328)、[作者 SPNet](https://github.com/houqb/SPNet) | 长条 H×1 / 1×W 分别聚合远距上下文。ADE20K ResNet-50 表2的两层 MPM：短程40.50、长程条带41.14、两者41.92；表3条带位置也会改变结果。不是 PIDNet 增益。RSM 的 PMPM 才有 PIDNet-S 的方向池化消融。 | 若道路/护栏等呈长轴错误，在同一 PAPPM 只换 scale3 作方形、条带和 PMPM 对照；保留 scale4 全局路径，再独立决定是否加 ECA。 |
| **RDRNet 的 RPPM**，Guoyu Yang、Yuan Wang、Daming Shi，2024 [作者全文 §III-C/§IV-F 图8](https://arxiv.org/html/2406.12496v1#S4.SS6)、[代码库](https://github.com/gyyang23/RDRNet) | 在 PAPPM 的训练路径添加并行分组卷积并折叠到部署卷积；RDRNet-S-Simple Cityscapes val 图8 RPPM 76.8，比该网络 PAPPM 高0.6 pp。是不同骨干的消融。GCNet 亦是独立网络，[原文 §3.2](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Golden_Cudgel_Network_for_Real-Time_Semantic_Segmentation_CVPR_2025_paper.pdf)。 | 若仍要求部署图接近原版，优先试针对 PAPPM 的 RPPM，其次才试改 BasicBlock。只在同一卷积几何且分支间可线性融合时声称可折叠；评估模式先折 BN，核 FP32 和目标精度的前后输出、边界 padding、导出以及设备测速。 |
| **训练期 EMA / 分组学习率** | [PyTorch AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html) 支持权重 EMA；注意其当前文档与本机 1.13 API 可能不同。仓库 [train.py](../../tools/train.py#L138) 用一个 SGD 参数组，只有两个组时 [调度器](../../utils/utils.py#L154) 才把第二组设为 nbb_mult 倍，默认10，不是这个倍率优于基线的证据。与 [EMA 注意力](https://arxiv.org/abs/2305.13563) 是两件事，后者会改变推理图。 | 以同训练轨迹比较在线权重与 EMA，保存 shadow/更新次数，明确 BN buffers 与恢复；分组 LR 先核每个参数只归一组、记录两组实际 LR，和 EMA 分开。不能把训练时间、显存或本项目提分视为零。 |

补充报告 §2.1 的名称可进一步落实：

- **AWA-PIDNet**：Guanhong Zhang、Hong Li，2025，[SPIE 13942 论文集题录](https://www.proceedings.com/content/082/082875webtoc.pdf)，1394226，活塞销表面缺陷。题录已核，摘要复载提示 I/P 的 AGCA、WTConv 融合及 ASPP；完整原表、PIDNet 变体、split、预训/TTA与作者代码未核，数字不作为道路证据。
- **PIDFormer**：Yanyan Chen、Huacai Chen，**2026-08-03** [SPIE 143050Z](https://doi.org/10.1117/12.3120274)。作者摘要是 BiFormer 双层路由注意力与混合注意力替代 Bag，具体插入层未据全文核实；自建火焰分割，作者摘要 mIoU 89.05，称高于原 PIDNet 1.89 个百分点，目标类 IoU 74.49/78.24。无原表/测速/split/预训/TTA或作者实现；不能记成 2024–25 的街景 +1.2～1.9。与煤流 **PID-Former** 是不同论文。
- **飞机部件论文**：Junqing Bai、Pan Yang，2025 ICSP pp.528–531，[IEEE 论文集题录](https://www.proceedings.com/content/081/081835webtoc.pdf)，[出版方登记 DOI](https://doi.org/10.1109/ICSP65755.2025.11086941)。仅核题录，未获可核原表，不采用原报告的 +1.8～2.3 pp，不纳入街景迁移候选。
- **MECS/ECA** 已对应 [Yang 等 MEP-Net 正式版](https://link.springer.com/article/10.1186/s12880-026-02416-x)：2026，PAPPM 后 MECS、Bag 后 ECA；原表是 CEUS/BUSI 超声，见[主表](report.md)。不另计一个街景模型。“精细边缘”条目对应李仲等的[水位线检测](https://www.ecice06.com/EN/Y2024/V50/I8/102)，mIoU限水线邻域。

## 代码草案：可保留的假设和需修的边界

| 原报告位置 | 代码和实测核查 | 实施时只改什么 |
| --- | --- | --- |
| §1.1 / EnhancedPagFM | [PIDNet.forward](../../models/pidnet.py#L144) 的 I 经 ReLU 后还过 [compression3/4 的 Conv+BN](../../models/pidnet.py#L41)，因此 PagFM 收到的 y 不保证非负。合成正输入经 Conv(-1)+BN，最小值 −0.999995。默认 [PagFM](../../models/model_utils.py#L268) 确用未缩放的逐像素内积和单通道 sigmoid；BN 并不保证独立同方差，也没有本地门控饱和统计。 | 先记录 pag3/pag4 logits 分位数、sigmoid 饱和比例、边界/内部和梯度范数。只对温度或 with_channel 做单改，保留原插值和激活作对照；不预设扩大通道门一定更好。 |
| §3 方案一的零偏移 | 2×2 斜坡到 4×4：草案 linspace(-1,1) 配 align_corners=False 和 border，零偏移与原 bilinear 最大差 **0.24999994**。用输出像素中心 2(i+0.5)/N−1，差 **0**。即使 offset=0，整个 EnhancedPagFM 还改了门控/激活。 | 将像素中心和偏移单位写清：位移若按输入格点，归一化用输入 H/W；若按输出像素，用输出 H/W。先用零偏移等价、非方形、边界测试隔离采样，再测实数据；模型内外 align_corners 已不同，统一是独立变量。 |
| §1.2 / Strip-PAPPM | 对 16×16 全1张量，17/8/8、含 padding 均值输出 **2×2：[0.280277, 0.498270; 0.498270, 0.885813]**，全局池化是1×1值1。窗口仍有不同有效覆盖，不能据核尺寸称退化或浪费1/5。草案 scale_process 仍是 groups=4 的 3×3 卷积，不能声称已替换成条带 depthwise。 | 只换尺度池化做对照；测多个输入尺度、窗口实际覆盖、融合后特征相关性和边界/内部指标。若还改分组卷积，另起消融。 |
| §1.3 / extract_valid_edge | [现实现](../../datasets/base_dataset.py#L106) 先对类别编号 Canny/膨胀再缩放裁剪；Void 污染与边宽变化有理由查，“所有交界均被标出”与精确边宽未实证。草案的两端有效条件能阻止直接生成 Void 边，但膨胀后 void 点从0变1，末行只屏蔽负例，实测无效扩张带里仍有16个正例。 | 明确规则只屏蔽 Void 像素，或屏蔽整条邻域带；在膨胀后对所选无效掩膜的正/负例都屏蔽，再由 loss 对有效像素计数及归约。若允许保留有效侧靠近 Void 的真实边界，不要一刀切扩大掩膜。先做非方图、裁剪、flip、全无效样例。 |
| §1.4 / BAS | [FullModel](../../utils/utils.py#L51) 用 D 概率>0.8 选标签；全 ignore 是既有已证风险，[前轮审计](../pidnet_accuracy_research/repo-audit.md) 有最小复现。硬条件不让 BAS 的选点梯度直接进入 D，但 D 还经 Bag 参与语义梯度，不等于 D 无梯度。 | 先让空掩膜分支返回有限且可反传的零，单测无正/无负/全无效；软门控、Dice、Lovász 是不同损失策略，逐项对照，避免把数值保护和提分混作一个结论。 |
| §1.5 / Dense 1/4、D-Guided PointRend | [D 最终特征及融合](../../models/pidnet.py#L168) 仍在1/8，可能只能引导已响应的位置；标准推理不返回辅助 D logits，使用它要额外保留边界头或明确从 D 特征如何得到分数。4C 门/融合张量与 C 的 layer1 投影不能直接逐通道相加；须投影到同通道数或设计单通道门，并让 final_layer 通道数匹配。 | 先区分粗输出漏检与边界错位；在同细化预算下比语义不确定性、D+不确定性，检查被采样的小物体召回。无局部证据时不承诺找回完整漏检，1/4 稠密头的代价单测。 |
| §1.7–1.8 / 训练配方 | 源码 CamVid 121轮停、200轮 poly 已核；[PIDNet 原论文及旧报告](../pidnet_accuracy_research/report.md) 写 CamVid 200轮调度和 LR 阈值早停，121轮不证明未收敛。两处 stem bias 的第三方权重缺失与前缀/加载覆盖另行追踪，不能把任一补救当提分。 | 分开记录复现配方与新日程，使用独立 val 选择，不靠 test 停步；先做权重覆盖、同初始参数与有效步数验收，再比较配方。 |

上面四个新探针可用 [脚本](../../.super-agent/29b3bc3410354c96ba50302baf031655-complement-probe.py) 和[输出](../../.super-agent/29b3bc3410354c96ba50302baf031655-complement-probe.txt) 复核。使用本机 segformer Python（Torch 1.13.0、OpenCV 4.1.2），只创建合成张量，没有导入或运行完整 PIDNet。

## 合并后的优先级

先固定数据、权重覆盖、有效标签/空 OHEM 和主头评估，分别做基线重测与修复对照。随后根据失败类型选一条路线：**区域内部错分**先 DenseASPP/PMPM，再单试 FADC、RPPM 或方向池化；**跨分支边缘错位**先测 PagFM 门分布和网格，单点 DySample/FreqFusion 其次；**边界噪声**先纠正 GT 和忽略策略；**小目标整块漏检**再看 1/4 特征与采样覆盖。EMA/分组 LR 是独立训练对照。任何方案的本地 mIoU、额外参数和延迟，在同一模型、数据和计时口径测到前均记为未知。

