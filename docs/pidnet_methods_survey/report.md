# PIDNet 改进方法调研：哪些论文真的改了 PIDNet

检索截止 **2026-11-11（北京时间）**。本项目以 Cityscapes / CamVid 的 RGB 街景精度为先，可以接受适度降速。

主表整理 **31 项**：21 项直接修改 PIDNet 结构或训练流程（含 3 项从补充报告复核的题录或摘要），6 项三分支衍生或结构近邻，4 项相关网络迁移。含原表量化结果的直接工作有 13 项；同一工作的预印本和正式版只算一次。没有本地 PIDNet 训练、推理或性能复现。

**第一轮建议优先做两组 I 分支单改：DenseASPP 和 PMPM，各自与同一 PIDNet-S/M 的 PAPPM 对照。** 两篇论文都有直接针对 PIDNet 的 Cityscapes 消融，单项分别从 **78.6 到 79.0（+0.4 个百分点）**、从 **78.3 到 79.1（+0.8，125→103 FPS）**。两个基线来自不同论文，不能用这两个差值给模块排序。FADC 是另一条直接证据：作者改的是 PIDNet-M 的 bottleneck 卷积，报告 **81.0 val、80.6 test、37.7 FPS**，但原 M 是引用参照，PIDNet 的具体配置也没有公开。它适合作为下一组单变量试验。

补充报告提供了 PagFM 对齐诊断、DySample/FreqFusion、条带池化、训练期 EMA 和 RPPM 这些方向。原论文数据、适用边界和草案修正放在[互补核查](complementary-review.md)。实际合成验证发现了零偏移网格误差和膨胀后的 void 正例泄漏，也否定了「大池化等于全局池化」「I 的 PagFM 输入必非负」两条推断。原报告给出的本地提分、毫秒和参数预估，不作为实测结论。

## 怎么读下面的表

表里的 mIoU 用百分数，速度用 FPS。每项都有一个证据等级：

- **A**：已核对原表，同一篇论文里有控制对照。
- **B**：已核对原表，但底模、引用或关键协议有局限，或者只能做不同模型之间的对比。
- **C**：只核到摘要/预览，或者没有可比结果。

等级是按具体实验算的，**域外的 A 不等于 Cityscapes 上的 A**。原文没有声明的评估 split、TTA、初始化和 FPS 条件，这里不补猜。数字和实现细节见[直接改造证据](direct-cases.md)、[上下文证据](context-cases.md)、[检索记录](search-log.md)。

## 直接修改 PIDNet：道路 RGB 与训练方案

| 工作 | 改了什么 | 结果、限制与建议 |
| --- | --- | --- |
| **Tan / Jin 2025**，[Improved PIDNet for Road Sensing](https://www.mdpi.com/2079-9292/14/5/871) | HRNet 式多分辨率特征提取；I 末端用 DenseASPP 替换 PPM；Bag 加交叉注意力。 | **A**。Cityscapes 表 3：PIDNet-S 78.6，只加 DenseASPP 79.0，三项全加 81.1；该消融没有标 split。表 1 是 val 78.8 / 81.0、test 78.6 / 81.1，RTX 3090、2048×1024、101/99 FPS；原 S 是引用值，不是同设重训。CamVid 表 2 是 80.1/82.6、160/155 FPS，split 和 TTA 未说明。正文 79.7 与表 3 的 HRNet+cross 79.9 冲突；表 4 的 16 类不能替代标准 19 类。没有定位到作者完整代码。建议先只改 I 末端，不要把 81.1 全归给 DenseASPP。 |
| **Zhang / Zong / Jiang 2025**，[RSM-Optimizer](https://www.mdpi.com/2079-9292/14/6/1109) | P 端加 RF-FEM；I 末端 PMPM 用方形、横条、竖条三路池化；SUFM 做分级上采样融合，含 CARAFE。 | **A**。Cityscapes 表 1 同一篇论文里：S 78.3/125 FPS，只加 RF-FEM 78.8/120，只加 PMPM 79.1/103，两项 79.4/98，全套 79.8/94；没有 SUFM 单项。表 3 CamVid 80.1/153 和 82.8/150，720×960；原表没有标 val/test、预训练/TTA 和精确计时条件。没有定位到作者代码。PMPM 值得首试；GFLOPs 接近不代表实际延迟接近，要单独实测。 |
| **Chen / Gu / Zheng / Fu 2024**，[FADC](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html) | 在 PIDNet 的 bottleneck 里使用频率自适应膨胀率、卷积核分量和频带权重，不是替换 PPM。 | **B**。正文表 3 和补充表 4：Cityscapes M-FADC 81.0 val / 80.6 test / 37.7 FPS，RTX 3090、2048×1024；原 M 80.1/80.1/39.8 是引用值，不是同设增益。补充材料写 120k、1024×1024 crop、SGD，ImageNet 和 TTA 没有按表核实。[作者算子](https://github.com/Linwei-Chen/FADC)只有通用 FADC 和 ADE20K 配置，没有 PIDNet 的具体替换层/配置；依赖 MMCV 可变形卷积和 FFT。 |
| **Chen / Ding 2025**，[Wavelet-EdgeNet](https://journals.sagepub.com/doi/pdf/10.3233/ATDE251140) | PIDNet-S 的 stem 改成多通道输入：RGB、DWT、SWT、自适应阈值 Canny；D 分支和边界标签没有被替换。 | **B**。CamVid test 960×720、367/101/233、ImageNet。表 1 原 S 重训 73.8/7.6M，完整 77.4/7.7M；表 2 单 SWT 76.46、单 Canny 73.33、全加 77.44。训练段写 120/200 epoch 不一致，每个 epoch 都看 test，选择规则和 TTA 未知，没有推理测速。没有定位到作者代码。边界失败时可以试 SWT 单项，但要算入变换开销。73.8 不能和官方 80.1 合并。 |
| **Wu 等 2026**，[唐卡 improved PIDNet](https://doi.org/10.1371/journal.pone.0332164) | P 的 Pag 加 ECA/残差；P 末端 LGFM；D 加 PConv3。 | **B**。Cityscapes 表 8 是 74.38/75.40，两行同为 80k、1024×1024 crop、RTX 3090；变体/split/预训练/TTA 未说明，512×512 是同表 Swin-B。**A（唐卡）**表 5：原 M 70.20→全套 73.33；单 LGFM 74.14，完整的边界 F 分数更高。[作者模型代码](https://github.com/JikD-591/thangka-seg/blob/main/mmseg/models/backbones/newpidnet.py)已打开。PConv3 实际是部分通道卷积，正文没有说学习掩码；街景配置是普通 PIDNet，不能复现表 8。ECA-Pag 可以作轻量候选。 |
| **Lian 等 2024**，[Modified PIDNet，EEI](https://ieeexplore.ieee.org/document/10696235) | 下采样 SOM；PAPPM 嵌入坐标聚合注意力；另加 PSA。 | **C**。原摘要复载称 Cityscapes 83.52；出版元数据已核，全文/原表/基线/变体/split/TTA 未取得。没有定位到作者代码；83.52 不参加排序，先索取正文再考虑复现。 |
| **Qian 等 2025 在线 / 2026 卷**，[PIDNet-LW](https://link.springer.com/article/10.1007/s11554-025-01799-4) | P 路卷积堆叠减少三分之一，重做 BasicBlock/Bottleneck。 | **C**。出版社摘要称参数少 47%，Aeroscapes 74.16、Cityscapes 76.7；原表需订阅，底模、split、TTA、预训练和测速不推断。没有定位到作者代码。只有压缩目标优先时才值得追。 |
| **Zhou 等 2026**，[PILOT 正式版](https://www.mdpi.com/2079-9292/15/13/2833) | 冻结底模，D 并联一个新类的二分类支路，通过置信门控合入预测。 | **A（增量）**。Cityscapes val 15→19 类，表 1 同 M：普通微调 59.90、仅新头 62.35、PILOT 71.87、联合上界 77.80。表 3 首个加类的 P/I/D 为 67.97/70.25/72.65；表 5 M 部署 71.9 FPS，A100、2048×1024 batch1。TTA/初始化没有按表确认。[作者 PIDNetCL](https://github.com/U1overground/PILOT/blob/main/models/pidnet_cl.py)已打开。正式标题是 Replay-Free；这不是普通 19 类全监督提分。 |
| **刘晓等 2023**，[JSRSSL 超分辨率联合训练](https://www.newit.ejournal.org.cn/zh/article/56265791/) | 共享 PIDNet 编码器，同时训练分割和超分，部署时移除超分支。 | **A（退化 RGB）**。表 1 CityDeg val：原域权重直测 22.50、同域重训 58.20、联合重训 62.15；后两行才可比。7.6M/11.8G，变体、TTA/预训练/FPS 没有细列。原始 Cityscapes 只是退化数据的来源。没有定位到作者框架代码。如果输入低清或压缩，值得验证；清晰街景不预期同等增益。 |
| **Sing Kuang Tan 2026**，[BS-PIDNet](https://zenodo.org/records/19496155) | 用布尔结构神经元技术改造卷积、归一化和注意力。 | **C**。作者 PDF 已读，没有 Cityscapes/CamVid 对照、mIoU 或运行协议。[作者 pidnet_sk.py](https://github.com/singkuangtan/BS_PIDNET/blob/main/models/pidnet_sk.py) 和 bsconv.py 已打开，普通 pidnet.py 不是改版；没有实测证据，低优先。 |

## 直接修改 PIDNet：专用图像、模态或领域

| 工作 | 改了什么 | 结果、限制与建议 |
| --- | --- | --- |
| **Xin 等 2026-11-02**，[Agri-PIDNet-S](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0361992) | 浅层 CA-MKIR；I 的 PAPPM 前加 SaE-D2T；末端 BWR-Bag 加 4×4 窗口注意力。 | **A**。表 1/4，同 4 类棉田 RGB test、495/107/107、1024×1024、20k。原 S 88.84→单项 91.71/91.66/91.79→完整 92.37；完整三种子 92.35±0.03。RTX 4090 FP32 batch1 前向 140.17→155.67 FPS。统一预训练规则、单次 resize。[作者模型](https://github.com/xyl-design/xinyanlin/blob/main/agri_pidnet/model.py)、[配置](https://github.com/xyl-design/xinyanlin/blob/main/configs/agri_pidnet_s.py)、[8 格消融 CSV](https://github.com/xyl-design/xinyanlin/blob/main/results/factorial_ablation_results.csv) 均已打开。可以借模块开关和消融设计，棉田结果不外推到道路。 |
| **Yang 等 2026**，[MEP-Net](https://link.springer.com/article/10.1186/s12880-026-02416-x) | I 路 PAPPM 后加中值增强空间/通道注意力 MECS，Bag 后加 ECA。 | **A**。表 1/3/5 私有 CEUS：77.71→单 MECS 77.84、单 ECA 78.56、合用 78.96；公共 BUSI：82.13→82.62、82.86、83.19。ResNet-18/ImageNet，200 epoch、RTX 3090，S/M/L 和 TTA 未明确；表 4 是 334.84→409.96 ms。[原表 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13339476/fullTextXML)已对读，没有定位到作者实现。只借注意力落点。 |
| **Cui 等 2025**，[Cherry-Net](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1607205/full) | 换 SwiftFormer-XS 和 SwiftRep-Hybrid，改末端 LFE 和损失。 | **B**。樱桃 RGB 1804/601/601，1312×1312 crop。表 4：原 S 70.19→换主干及减损失 71.64→SwiftRep 72.00→LFE 72.20；表 4/6 注 val，§2.7 写 test，不确定以哪套为最终；表 7 同 TX2/TensorRT FP16 是 10.3→10.8 FPS。没有定位到作者代码；可以借局部/全局融合设计，整套换主干成本大。 |
| **李仲等 2024**，[河流水位线 PIDNet](https://www.ecice06.com/EN/Y2024/V50/I8/102) | I 路下采样加 LPEM，池化换 LCPPM。 | **A**。自建河流 RGB 960×720，表 3/4 原 S/M 87.96/90.31→联合 89.04/91.35，M 38.9→38.6 ms（Tesla A30）。mIoU 只算水线上下 60 像素带，不能和全图街景比。没有定位到作者代码；可以看简化池化与通道权重。 |
| **李嘉豪等 2025**，[EMA-PIDNet 玉米倒伏](https://www.j-csam.org/jcsam/article/abstract/20251136) | RGB+DSM 四通道，I 主干和 I→P/D 各接 EMA。 | **A**。表 2/3，3 个生育期，RGB 基线 74.09/72.84/72.61；加 DSM 79.61/78.70/76.87，DSM+三处 EMA 83.94/82.77/82.64；I 单 EMA 均下降，没有 RGB-only EMA 行。地块独立测试，512×512，预训练/TTA 未知。没有定位到作者代码。不能把高度模态的增益算作 RGB attention。 |
| **Kimura 等 2025**，[Pixel-weighted Domain Adaptation](https://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P572.pdf) | Cityscapes 预训练 M，源/目标两个头，像素/类权重和域判别器。 | **A**。农路 RGB 10 类，481/399 按地点分开，1024×512/15k。表 I：单目标 26.43、朴素双域 37.91、加权 38.38、再加判别器 40.26；加熵反而降到 32.50。没有 TTA。没有定位到作者代码。只在域迁移、标签偏斜场景有参考价值。 |
| **Zheng 等 2026**，[UAV imagery improved PIDNet](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6271048) | 重做 I 的 PAPPM 和轻量语义头。 | **C**。预印本摘要称 UAVid 70.73/71.30、改头 72.33，Caihua Village 87.27/89.17、改头 89.92；正文 403，原表和各行协议未核。没有定位到终版或作者代码；航拍域、参数降幅只作线索。 |
| **Li 等 2026**，[DGHF-PIDNet](https://link.springer.com/article/10.1007/s11554-026-01903-2) | P 加方向卷积 DAGRConv，HF-BIGate 替代 P/I 交互。 | **C**。付费摘要写 SemanticRT/FMB/PST900，RGB-T，称 7.85M 和 218 FPS；原表、split、预训练/TTA 和测速硬件未核。动态门控如何折叠未证。作者称合理请求可提供代码，没有定位到公开实现。单目 RGB 不按该数估计增益。 |
| **Chen / Chen 2026**，[PIDFormer](https://doi.org/10.1117/12.3120274) | BiFormer 路由注意力、混合注意力替代 Bag。 | **C**。SPIE 摘要，自建火焰分割 mIoU 89.05，作者称高于基线 1.89 个百分点；全文原表、底模、评估设置未核。没有定位到代码；与下面煤流 PID-Former 不是同一篇，不作为道路实验。 |
| **Zhang / Li 2025**，[AWA-PIDNet](https://www.proceedings.com/content/082/082875webtoc.pdf) | 活塞销缺陷，摘要复载给 AGCA、WTConv、ASPP。 | **C**。SPIE 13942:1394226 题录已核，全文原表、协议未得。没有定位到代码，保留为工业域线索。 |
| **Bai / Yang 2025**，[飞机部件 improved PIDNet](https://www.proceedings.com/content/081/081835webtoc.pdf) | ICSP pp.528–531，DOI 10.1109/ICSP65755.2025.11086941。 | **C**。题录核实，原表与同设基线未得，不采纳补充报告的增益。没有已核作者实现，不列为道路迁移候选。 |

## 三分支衍生与其他可迁移方法

| 工作 | 改了什么 | 结果、限制与建议 |
| --- | --- | --- |
| **任凤雷等 2026**，[三分支 EPPM / MCIAM](https://ope.lightpublishing.cn/zh/article/doi/10.37188/OPE.20263401.0167/)（衍生） | 保留 Bag、取消 Pag；I 端逐级池化加门控；P/D 的 MCIAM 空洞卷积沿**通道序列**。 | **A**。表 2 作者模型 Cityscapes：PAPPM 78.7/98.2→EPPM 79.2/88.5，ASPP 78.9/79.3，DAPPM 79.0/92.3；表 3 无注意力 78.7→MCIAM 79.2。表 5/6 作者 79.2/88.5 与 80.5/140.1，RTX 3090，不是原 PIDNet 重训。表 1 有伪代码，没有可核的完整作者库。 |
| **Zhao 等 2026**，[TBMSNet](https://www.nature.com/articles/s41598-026-48759-x)（衍生） | 保留 Pag；I 用 DWR 空洞深度卷积和 MSAPP。 | **A**。表 8 Cityscapes val：PAPPM 80.1/51.1→MSAPP 80.5/50.4；表 5 固定 SIR+MSAPP 加 DWR，79.1/78.3→80.5/50.4。表 3 test 80.5/50.4，RTX 3080，单尺度、2048×1024、ImageNet、train+val；CamVid 表 2 的 86.1/165.2 只有作者值，引用 PIDNet 型号误标，类映射和 split 未核，不算 +6。代码可请求，未见公开。 |
| **Ren 等 2026**，[ARFNet](https://doi.org/10.1016/j.image.2026.117640)（衍生） | HDAPM 多率稠密空洞池化，TDIM 行/列/全局注意力。 | **C**。预览写 Cityscapes 79.8/94.8、CamVid 80.9/154.1、RTX 3090，原表/协议/代码未得。与上面同团队的 EPPM 文是不同研究。 |
| **Shi 等 2025**，[HFINet](https://doi.org/10.1016/j.image.2025.117375)（三分支近邻） | 边界支 ABPRM 异尺度注意力，末端 EAPPM。 | **C**。引言写 78.0/63.5、5.3M、1024×2048，split/表/硬件/代码未核；不是下一条 TSNet。 |
| **Wang 等 2026**，[TSNet](https://doi.org/10.1016/j.dsp.2026.106278)（三分支近邻） | U 形空间/边界支、语义 EAPPM、边界 BGFM。 | **C**。TSNet-T 摘要写 Cityscapes 78.7/133.4、CamVid 79.4/208.7，表、模型配置和代码未得。 |
| **Xu 等 2025**，[PID-Former 煤流](https://doi.org/10.21203/rs.3.rs-6607533/v1)（衍生） | I 轻量 Transformer，P/D 倒残差，CIFM/EFM 补上下文，SEB 细化边缘。 | **B**。[预印本原表 II](https://assets-eu.researchsquare.com/files/rs-6607533/v1_covered_b1b7f6e6-30e3-4dc2-83fd-5a4e73a2cdd4.pdf?c=1747652778) 自建煤流 test：S 95.68/110.9 与 96.99/204.1，RTX 3090；TX2 40.3/67.1。没有终版/作者代码，专用域。 |
| **Yang 等 2025**，[APTNet](https://doi.org/10.1016/j.jvcir.2025.104594)（迁移） | 受 CIDNet 启发，不是 PIDNet 改版。多层空洞卷积、BARM 双向交互、RPAB 细化上下文。 | **C**。预览 Cityscapes val：Slim 75.9/85.1、Base 79.9/41.2；[作者仓库](https://github.com/Yanghaotian-CV/APTNet)已打开，只有占位。 |
| **Cărunta 等 2026**，[RTSSNet](https://www.nature.com/articles/s41598-025-29931-1)（迁移） | 单路编码末两级空洞卷积，解码借 Bag、没有独立 D。 | **A（作者网）**。CamVid 消融：后两级 75.02、普通残差 74.95，早加空洞更慢。Cityscapes val 78.3 / CamVid test 77.53；CamVid 数据链接指向 Kaggle，无法确认与 PIDNet 同 split。作者代码未确认。 |
| **Gao 2023**，[RegSeg](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Gao_Rethinking_Dilated_Convolution_for_Real-Time_Semantic_Segmentation_CVPRW_2023_paper.pdf)（迁移） | 分组卷积把通道分成小/大 dilation，静态防漏格参考；[作者实现](https://github.com/RolandGao/RegSeg/blob/main/blocks.py#L62)已打开。 | **B**。Cityscapes 表 3 固定解码器：RegNetY-600MF 75.66、ASPP 77.29、PPM 77.47、换 RegSeg 78.65，最后一项是主干对照。CamVid test 80.9±0.07，T4 混精度、batch4，不与 PIDNet batch1 速度混算。 |
| **Lv 等 2026**，[DSAFNet](https://www.mdpi.com/2079-9292/15/4/866)（迁移） | 编码器末端 MDFA：分解空洞卷积加注意力。 | **A（作者网）**。表 9 Cityscapes val 73.0→只空洞 73.95→加注意力 74.78，RTX 3090、1024×512；不是 PIDNet，没有定位到作者代码。 |

早期机制可以追溯到 [DenseASPP（CVPR 2018）](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html) 和 [HDC/DUC（2017）](https://arxiv.org/abs/1702.08502)。原 [PIDNet CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) 表 3 里，S 在同样使用 Light-Bag 时 DAPPM 和 PAPPM 的 mIoU 相同，PAPPM 更快，所以原论文不支持「默认升级池化」。大核或结构重参数可以参考 [GCNet（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Golden_Cudgel_Network_for_Real-Time_Semantic_Segmentation_CVPR_2025_paper.pdf) 的部署折叠；[RDRNet 图 8](https://arxiv.org/html/2406.12496v1#S4.SS6) 的 RPPM 更接近 PAPPM 的接口。这两者都是独立网络。边界融合/训练可以参考 [BFMSNet](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70287) 和 [SCTNet](https://arxiv.org/html/2312.17071)；对照条件沿用[前轮研究](../pidnet_accuracy_research/report.md)，不计入直接工作。

## 在本仓库怎么试

1. **上下文失分时，先换 I 末端一处。** 固定 S 或 M，在[构造的 `self.spp`](../../models/pidnet.py#L84) 上分别做 PAPPM→DenseASPP、PAPPM→PMPM，保持通道、输出尺寸、P/D、Pag、Light-Bag 和辅助损失不变；L 的 DAPPM + Bag 另立容量对照。EPPM、MSAPP 或 RPPM 可以作后续 I 端候选。1024×1024 裁剪到 1/64 只有 16×16，别照搬在高分辨率特征上选的 dilation；保留小率或池化/残差路径，核查 padding 与采样覆盖。先对照固定小率，再判断额外注意力值不值。

2. **有结果后，再单独试 FADC 或静态分组空洞。** 保持 PAPPM，工程起点可以选 M 的 I-layer5 后一个 stride=1 的 3×3 `conv2`；这是本仓库的建议，论文没有披露具体层。FADC 要先确认 MMCV/可变形算子、FFT、FP16 与部署后端，优先对照固定膨胀率和 RegSeg 式分组，再量延迟。参数或 FLOPs 接近，不等于实际延迟接近。

3. **边界/细物体失分时，先查 P/D 再换融合。** 先区分标签噪声、对齐错位和完全漏检：检查边界真值和 PagFM 门分布，如果确实错位，单点对照 DySample/FreqFusion；如果门控不足，小步测 ECA-Pag 或 D→Bag 前注意力，保持 D 的辅助监督。看窄边带和小目标 IoU，再考虑 HRNet、交叉注意力或 Wavelet 辅助输入。只在类别递增或输入退化时试 PILOT、JSRSSL；权重 EMA 与分组 LR 单独测试。

4. **固定对照协议。** 用同一干净基线、训练/验证划分、权重初始化和训练预算，固定单尺度、不翻转；Cityscapes 19 类先 val 后 test，CamVid 固定 11 类、同 split 与 Cityscapes 初始化来源。在同设备同输入 batch1 下记录 mIoU、小目标/窄边带、前向与端到端 p50/p95、显存；上线前再测导出环境。只在单项胜出后组合；[旧基线检查与实验设置](../pidnet_accuracy_research/experiment-plan.md) 仍然适用。

---

> 修订记录：2026-11-11，ZCode（opencode-go-chat/deepseek-v4.1-flash）重写全文，调整表格与表述，保留全部条目和数字，并核对链接。原始结论与数据来源未改，按当前代码修正了少量表述。
