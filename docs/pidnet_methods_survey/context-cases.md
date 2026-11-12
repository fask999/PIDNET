# PIDNet 的空洞卷积、多尺度上下文与三分支案例

核查日期：2026-11-11（Asia/Shanghai）。任务是 RGB 街景语义分割，优先 Cityscapes、CamVid 的精度，允许适量延迟。结果格式为 mIoU（%）/ FPS，差值用百分点。同论文的同设消融可支持模块增量；带参考编号的横向比较仍是引用值。论文中没有清楚交代的测试协议保留为未知。未运行训练或测评，也未下载权重。

## 1. FADC：直接用于 PIDNet-M，作者公开了算子

Linwei Chen、Lin Gu、Dezhi Zheng、Ying Fu，*Frequency-Adaptive Dilated Convolution for Semantic Segmentation*，[CVPR 2024 正式发表版](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html)，2024 年 6 月（arXiv 首发 2024-03-08，预印本作者栏无 Dezhi Zheng）。[CVPR 正文](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.pdf)、[作者预印本](https://arxiv.org/pdf/2403.05369)、[CVPR 补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Frequency-Adaptive_Dilated_Convolution_CVPR_2024_supplemental.pdf)、[作者报告幻灯片](https://cvpr.thecvf.com/media/cvpr-2024/Slides/30701.pdf)。

- 作者正文 §3、§4.1（会议版第 3418 页）：AdaDR 根据局部频率预测空间可变的膨胀率；AdaKern 按通道重配卷积核的低频均值和高频残差；FreqSelect 通过空间权重重配输入频带。PIDNet 的描述只到“替换 bottleneck 的 convolution”，未细分 I/P/D、哪一层、是否替换带 stride 的卷积。它不是 PAPPM/DAPPM 替换实验。
- 正文表 3 与补充表 4：Cityscapes RGB，作者 PIDNet-M-FADC **val 81.0 / test 80.6 / 37.7 FPS**，34.6M、198.4G；RTX 3090、2048×1024。表中 PIDNet-M [76] 的引用结果是 80.1 / 80.1 / 39.8 FPS，34.4M、197.4G；两行表差为 +0.9/+0.5 个百分点、−2.1 FPS，未找到原版 M 同设重跑的证据。PIDNet-L [76] 为 80.9 / 80.6 / 31.1 FPS，作者模型在 test mIoU 持平。补充材料 §F 记载 PIDNet 120k 次、初始 LR 0.01、weight decay 5e−4、1024×1024 训练裁剪、batch 12、SGD、poly 和训练时缩放/翻转；额外分割数据预训未标 †，其 ImageNet 初始化、TTA/翻转、测试配置细节不能独立核验。未见 CamVid、S/L 型号或 PIDNet 上 AdaDR/AdaKern/FreqSelect 单项消融。
- 打开[作者代码](https://github.com/Linwei-Chen/FADC)并核对 `923102e` 树：[FADC 独立算子](https://github.com/Linwei-Chen/FADC/blob/923102e6344d9b90ec427cb15201efd514bf316e/FADC_only/conv_custom.py#L340)调用 MMCV modulated deformable conv，含 FFT 频带实现；[公开配置](https://github.com/Linwei-Chen/FADC/blob/923102e6344d9b90ec427cb15201efd514bf316e/configs/FADC/upernet_R50_FADC_512_160k_ade20k.py)是 ADE20K UPerNet-R50。未见 PIDNet 文件、配置或权重；[作者仓库询问 PIDNet 的 issue #16](https://github.com/Linwei-Chen/FADC/issues/16)无回复。不能从 ADE20K 配置推定论文 PIDNet 的准确落点。

## 2. 任凤雷等：EPPM 与 MCIAM，保留 Bag、取消 Pag

任凤雷、高紫阳、张炎、周海波、杨璐、秦志昌，《基于三分支网络的实时图像语义分割》，《光学精密工程》2026 年 1 月，34(1):167–177。[期刊和 DOI](https://ope.lightpublishing.cn/zh/article/doi/10.37188/OPE.20263401.0167/)、[期刊全文 PDF](https://ope.lightpublishing.cn/rc-pub/front/front-article/download/146561864/lowqualitypdf/Segmentaci%C3%B3n%20sem%C3%A1ntica%20de%20im%C3%A1genes%20en%20tiempo%20real%20basada%20en%20red%20de%20tres%20ramas.pdf)。

- §3.1–3.3、图 2–4、表 1：沿用 PIDNet 的细节/上下文/边缘角色与最终 **Bag**，为减开销明确取消 **Pag**。I 末端 1/64 的 EPPM 对多尺度平均池化做逐级残差聚合，拼接压缩后用 GAP 门控，再与输入投影相加。P 和 D 用 MCIAM：对空间 GAP 后的**通道序列**做五路并联 1D、kernel=3、dilation=1/3/5/7/9，求和后 sigmoid 重标定；这些 dilation 不是图像平面空洞卷积。表 1 有 PyTorch 风格伪代码，未检到完整作者仓库或训练配置。
- Cityscapes RGB §4.4，作者同论文表 2（第 173 页）：无池化 76.2 / 103.5、ASPP 78.9 / 79.3、DAPPM 79.0 / 92.3、PAPPM 78.7 / 98.2、**EPPM 79.2 / 88.5**；EPPM 相对 PAPPM +0.5 个百分点 / −9.7 FPS，相对 DAPPM +0.2 / −3.8。表 3：无注意力 78.7 / 93.1，SE 78.9 / 85.2，CBAM 79.0 / 82.8，ECA 79.0 / 90.6，MCIAM 79.2 / 88.5。表 4：Add 78.7 / 96.6，Concat 78.9 / 95.7，Bag 79.2 / 88.5。这些增量属于作者自己的网络和各自消融，不能相加。§4.4 未为消融表明确标 val/test。
- 横向表 5、6（第 174–175 页）：作者 Cityscapes **79.2 / 88.5**（2048×1024），CamVid **80.5 / 140.1**（§4.1 数据为 960×720，表 6 未另列 FPS 输入）。平台 RTX 3090、CUDA 11.2；PIDNet [23] 引用值分别为 S 的 78.6 / 93.2 与 80.1 / 153.7。后两组引用是原 PIDNet 的 test 数字，本文表 5、6 未明确作者行的 split、TTA 或预训练，不能据此声称同设重跑或跨数据集普遍增益。

## 3. TBMSNet：保留 Pag 的三分支衍生，MSAPP 有直接消融

Yuefeng Zhao、Guicong Zhang、Nannan Hu、Zhiqiang Shen、Yanxuan Wang、Yuxin Song，*A triple-branch multi-scale network for real-time semantic segmentation*，Scientific Reports 16:18124，2026-04-19。[出版社正文](https://www.nature.com/articles/s41598-026-48759-x)、[原版 PDF](https://www.nature.com/articles/s41598-026-48759-x.pdf)、[PMC 全文镜像](https://pmc.ncbi.nlm.nih.gov/articles/PMC13253871/)。

- 独立改造三分支，非仅换一个 PIDNet 模块。正文结构图、表 1、式 (3)–(5)：用 CustomStem、SIR 提取早期特征；I 侧用 DWR 分区特征、并行多率深度空洞卷积（正文举 d=3/5）和 1×1 融合，将上下文提取到 1/64，再用 MSAPP（式 (4) 为并行多率深度卷积、拼接、1×1 投影与残差；不要仅从模块名推成普通平均池化）。细节支明确采用 PIDNet 的 Pag，语义特征也逐点加至边缘支，最后融合。正文 Code availability 说向通讯作者按合理请求索取，搜索未见公开仓库或复现实验配置。
- Cityscapes RGB **val** 同设消融：表 8 PAPPM **80.1 / 51.1 → MSAPP 80.5 / 50.4**（+0.4 个百分点、−0.7 FPS）。表 5 为 TBMSNet 内部基线（不是 PIDNet）：77.2 / 93.2，仅 SIR 77.8 / 89.3，仅 DWR 78.7 / 67.5，SIR+DWR 79.0 / 61.7，SIR+MSAPP 79.1 / 78.3，全体 80.5 / 50.4；固定 SIR+MSAPP 看 DWR 为 +1.4 个百分点 / −27.9 FPS。表 7 的 DWR 1/32、1/64、1/128 分别为 78.3 / 62.6、80.5 / 50.4、79.1 / 46.2，不能推断 PIDNet 改同样分辨率会有相同增益。
- Cityscapes **test** 表 3：作者 TBMSNet **80.5 / 50.4**、22.9M / 85.3G；无 DWR 的 TBMSNet-S **79.1 / 78.3**、15.3M / 54.9G，单尺度、2048×1024、RTX 3080。表中引用 PIDNet-M **80.1 / 39.8**在 RTX 3090，FPS 不同硬件；其 FLOPs 表作 147.4G，与 PIDNet 原文和 FADC 的 197.4G 不同，不能用作精确复杂度差。作者称先 ImageNet 预训，Cityscapes test 模型用 train+val，CamVid 用 Cityscapes 权重微调；推理单尺度，未明确是否 flip。
- CamVid 表 2：作者 **86.1 / 165.2**（正文数据分辨率 960×720，实验平台 RTX 3080；表未单列测时输入或 split，§Experiment 语境为基准测试）。论文把 PIDNet-S-Wider 的引用行写成 80.1 / 153.7，这对数字在 [PIDNet CVPR 正文](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.pdf)表 5 属于 PIDNet-S，而 S-Wider 为 82.0 / 85.6；TBMSNet 正文又把 CamVid 数据写为 32 个标注类别，未清楚交代 86.1 的类别映射或公布 split 文件。将 86.1 留为作者表值，不能当作对 PIDNet 同协议的 +6.0。

## 4. ARFNet：明确由 PIDNet 的不足出发，稠密空洞池化候选

Fenglei Ren、Ziyang Gao、Haibo Zhou、Lu Yang、Xin He、Muyu Li，*ARFNet: Attentional refocusing fusion network for real-time semantic segmentation of urban road scenes*，Signal Processing: Image Communication 148:117640，2026 年 10 月。[出版社预览](https://www.sciencedirect.com/science/article/abs/pii/S0923596526001633)、[DOI](https://doi.org/10.1016/j.image.2026.117640)。与案例 2 多位作者重合，模块名和数值不同，不能把两个实验当作一组。

- 期刊引言明确针对 PIDNet 的细粒度与多尺度融合不足；采用三分支。HDAPM 以多膨胀率稠密连接、显式降通道、残差与预激活替代标准金字塔池化；TDIM 对全局、行和列进行跨维注意力。预览未证实 TDIM 所有分支的准确插入点，也未能核对 Pag/Bag 的保留与否。
- 期刊摘要/引言作者值：Cityscapes RGB **79.8 / 94.8**，CamVid RGB **80.9 / 154.1**，单 RTX 3090。未获全文和作者仓库，原表、基线重跑、split、输入、TTA、预训练、单模块增量均未核。优先补证 HDAPM 的稠密连接及模块消融，再决定是否按论文完整移植。

## 5. APTNet：借鉴其他三分支思想，非 PIDNet 改版

Haotian Yang、Yongsheng Dong、Lintao Zheng，*APTNet: Attention-refined progressive three-branch network for real-time semantic segmentation*，JVCIR 113:104594，2025 年 12 月。[期刊预览](https://www.sciencedirect.com/science/article/abs/pii/S1047320325002081)、[DOI](https://doi.org/10.1016/j.jvcir.2025.104594)。

- 引言称设计受 **CIDNet** 启发，PIDNet 用来说明多分支有效性。APTNet 五阶段逐渐分出三个空间尺度；basic block 和末端 deep aggregation block 用多层空洞卷积。中间阶段的 BARM 双向交换信息，低分辨率支通过 RPAB 细化全局上下文，不能把三个尺度直接等同于 PIDNet 的 P/I/D 三种职责。
- 作者引言公开 Cityscapes RGB **val**：Slim **75.9 / 85.1**、Base **79.9 / 41.2**（无加速技巧）。确认也评估 CamVid，预览不含具体分数、正式表格、作者内控基线、输入、TTA、预训或硬件。实际打开[论文列出的仓库](https://github.com/Yanghaotian-CV/APTNet)，只有一条提交和占位 readme，无实现或配置。BARM 可作 I→P 交互参考，暂无 PIDNet 插件证据。

## 6. HFINet 与 TSNet：I 末端池化与边界注意力的独立三分支参考

- **HFINet**：Jixiang Shi、Jin Liu、Wen Lu、Ruisen Liu、Jiajun Wang，*Heteroscale feature integration network for real-time semantic segmentation*，Signal Processing: Image Communication 138:117375，2025 年 10 月。[出版社预览](https://www.sciencedirect.com/science/article/abs/pii/S0923596525001213)、[DOI](https://doi.org/10.1016/j.image.2025.117375)。引言讨论 PIDNet 的 D 支，作者架构同样有语义、细节、边界支，未找到基于 PIDNet 源码或基线的证据。§3.2 边界支串联 ABPRM，从异尺度语义特征中经空间和通道注意力细化边界；§3.4 EAPPM 从低分辨率输入提取并汇聚多尺度上下文。期刊引言给 **78.0 / 63.5**、5.3M、1024×2048；§4 预览确认 Cityscapes/CamVid RGB 都评测，未核 78.0 对应 split、CamVid 数字、原表、硬件、TTA 和预训。RCNet 是引言横向比较。未找到全文可见表或作者代码；页面 *Cited by* 的 **78.7 / 133.4、79.4 / 208.7** 属于下条 TSNet，不得归给 HFINet。
- **TSNet**：Junxing Wang、Wenbin Zou、Jie Su，*TSNet: Three-branch scale-aware network for real-time semantic segmentation*，Digital Signal Processing 181:106278，2026-09-15。[出版社预览](https://www.sciencedirect.com/science/article/pii/S1051200426003969)、[DOI](https://doi.org/10.1016/j.dsp.2026.106278)。参考 PIDNet 设语义、空间、边界三支，公开预览未证明直接修改 PIDNet。空间支 USM、边界支 LUSM 用 U 形跨尺度提取，语义支末端 EAPPM 渐进融合多尺度上下文，BGFM 用边界指导语义与空间融合。作者摘要 TSNet-T：Cityscapes RGB **78.7 / 133.4**，CamVid RGB **79.4 / 208.7**。预览指 Cityscapes **消融**用 val；摘要性能对应 split、消融数字、输入、TTA、预训、硬件、基线和作者代码未核。模块同名不代表与 HFINet 实现相同。

## 7. RTSSNet：复用了 Bag 思路，后两级空洞卷积有位置消融

Cristina Cărunta、Alina Cărunta、Alexandru-Eugeniu Mizeranschi、Călin-Adrian Popa，*Real-time semantic segmentation of driving scenes via effective attention-based information fusion and hybrid encoder*，Scientific Reports 16:2419，2026-01-17。[出版社全文](https://www.nature.com/articles/s41598-025-29931-1)、[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819412/)。

- 单路编码器，非 PIDNet 三分支改版。末两级 1/16、1/32 用由 ANet/ASPP 启发的空洞块（d=6/18/24）；解码器复用 PIDNet Bag 思想，两组跨阶段特征融合以高分辨率特征 sigmoid 作门，未用独立 D 支。原文图 4–5、式 (1) 可复核；搜索只见第三方实现，未认定为作者代码。
- 表 2、4：RTSSNet-L Cityscapes RGB **val 78.3 / 121.15**，CamVid RGB **test 77.53 / 120.68**，RTX 3090，作者 FPS 分辨率 Cityscapes 1024×512、CamVid 960×720，CamVid 用 Cityscapes 权重。表中 PIDNet 等开放方法的 FPS 有作者在统一环境重测，精度来源不能据此都视为重训。更关键的是 RTSSNet *Data availability* 明列 Kaggle 的 carlolepelaars/camvid；[PIDNet 作者 README](https://github.com/XuJiacong/PIDNet#datasets)特意警告该 split 不用于其 CamVid 成绩。没有实际 split 文件对照时，保留作者数字但不跨论文算精度增量。
- 作者 CamVid test 消融 960×720，表 6、7 采用 ImageNet 预训：全用普通残差的编码器 74.95，后两级用空洞块的最终小编码器 75.02；从第 2 级就开始用空洞块为 75.00 / 191.16，对比后两级 75.02 / 244.60。固定小编码器的无 Bag 解码 69.31 / 312.72，两个 Bag 的最终解码 75.02 / 244.60；对照同时加入了解码与多尺度输入，不能拆成单个 Bag 的净增益。表 8 的膨胀率实验无预训，不能与表 6/7 直接拼差。

## 8. 通用迁移对照：RegSeg、DSAFNet

- **RegSeg**：Roland Gao，*Rethinking Dilated Convolution for Real-Time Semantic Segmentation*，CVPR Workshops 2023 年 6 月。[CVF 正文](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Gao_Rethinking_Dilated_Convolution_for_Real-Time_Semantic_Segmentation_CVPRW_2023_paper.pdf)、[作者实现](https://github.com/RolandGao/RegSeg)、[分组空洞卷积源码](https://github.com/RolandGao/RegSeg/blob/main/blocks.py#L62)。非 PIDNet；在 RegNetY 风格主干的 3×3 分组卷积内部切分通道，一组保留小 dilation，一组扩感受野，论文给定避免支持集漏格的膨胀率上界；代码以标准 Conv2d、split、concat 实现。Cityscapes RGB 表 3 同设固定解码器：RegNetY-600MF 无上下文模块 75.66±0.53 / 31、加 ASPP 77.29±0.18 / 28、加 PPM 77.47±0.29 / 30、RegSeg 主干无额外模块 78.65±0.28 / 30。这是不同主干对照，非单层插拔。表 7–8：无 ImageNet 预训 Cityscapes **val 78.50、test 78.3 / 37**；同一作者模型加入 ImageNet 预训 **val 79.42、test 79.1 / 37**；CamVid **test 80.9±0.07 / 112**，从 Cityscapes 预训初始化。§4.6 是 T4 混精度，Cityscapes 2048×1024/batch 1，CamVid 960×720/batch 4；不能直接拿 FPS 与 PIDNet RTX 3090/batch 1 比。未核 TTA。适合作为 FADC 的静态空洞、无可变形算子对照。
- **DSAFNet**：Wencong Lv、Xin Liu、Jianjun Zhang、Dongmei Luo、Ping Han，*Dilated–Separable Convolution and Attention Fusion Network for Real-Time Semantic Segmentation*，Electronics 15(4):866，2026-02-19。[期刊全文](https://www.mdpi.com/2079-9292/15/4/866)。非 PIDNet，独立不对称编码/解码器；MDFA 在编码器末端用分解空洞卷积和通道/空间联合注意力；DL-MBDC 是编码器深度卷积块，MALD 是注意力解码器。Cityscapes RGB **val** 同设表 9（RTX 3090、1024×512）：去 MDFA 73.0 / 76.48，仅多尺度空洞融合 **73.95**（文中 +0.95），完整 MDFA **74.78 / 74.74**（再 +0.83）；表 11–13 报作者 Cityscapes **test 74.78 / 74.74**、CamVid **test 70.5 / 89.5**，1.00M。不能把低宽度模型的提升归为 PIDNet 增量。未核 TTA、对应预训练，未见作者代码。若需对照注意力的额外贡献，可在本仓库同一个 I 端空洞模块上做去/加注意力消融。

## 按本仓库边界接入：工程建议

这些是按当前代码推得的候选实验；论文结果不能外推到这些变体。[models/pidnet.py](../../models/pidnet.py) 的 S/M 使用 PAPPM + Light_Bag，L 使用 DAPPM + Bag。I 分支 `layer5 → spp → resize(1/8)` 才送入 `dfm(P,I,D)`；P 在 1/8 经两次 PagFM 收到 I 支中间特征；D 有独立边界辅助头。保持边界监督、预测头和 logits 尺寸是分离模块因果的前提。

| 实验位置 | 建议先验证 | 保留的约束 |
| --- | --- | --- |
| I 支末端 `self.spp`，默认第一个池化实验 | S/M：PAPPM 对 EPPM、MSAPP，分开训练；L 保持原 DAPPM 作对照。DenseASPP/HDAPM 作第二轮，HDAPM 尚需原表和结构。 | 现有 `spp` 输入 `planes*16`、输出 `planes*4`，同低分辨率；统一上采样至 1/8。在 1024×1024 裁剪时 1/64 只有 16×16，大 dilation 大量命中 padding，不能照搬别网的率；保留近邻与池化路径，核实际采样覆盖。 |
| I 支 `layer5` bottleneck 空洞卷积 | M 的 `layer5` 后一块 stride=1 的 3×3 `conv2` 首试 FADC，同时试普通、固定 dilation 或 RegSeg 式小/大 dilation 分组。第一块 stride=2 的替换另测，P/D 的 bottleneck 另列。 | 这是保守起点，非作者公开实现。保留通道、残差、stride、padding 和原 PAPPM。测 MMCV 算子、FFT、混精度和实际推理延迟；部署端如无法支持，自定义算子的优势不能只按 FLOPs 推断。 |
| P 支 / PagFM、D 支 / Bag | 先给 P 单独加入 MCIAM；D 单独一组，最好先放最终 D 特征到 Bag 之前，另测对辅助边界头之前的影响。保留 Pag 作为第一轮对照，取消 Pag 独立消融。ABPRM、TDIM、BGFM 等等拿到更完整结构再做。 | 不把通道序列的空洞率视作像素空间 dilation；保持 P/D 为 1/8，D 输出的训练监督和融合门分布可比较；记录整体 mIoU 与 boundary IoU、小物体类别 IoU，以及同一设备 batch 1 延迟。 |

Cityscapes 从同版 PIDNet-M、同初始化/seed 和同一训练裁剪启动，固定单尺度、关闭 flip TTA 作主对照，val 筛选后才做 test；CamVid 固定 11 类、同 split 文件和 Cityscapes 预训来源。池化与 FADC 分别通过后才做组合，确认改变的分支与延迟可归因。

## 扩搜与排除

实际使用的查询包括 `FADC PIDNet Cityscapes bottleneck config`、`FADC PIDNet github issue`、`"PIDNet" "dilated" "PAPPM" Cityscapes CamVid`、`"PIDNet" "Cityscapes" "CamVid" "improved" dilated convolution attention pyramid pooling`、`"TBMSNet" Zhao Zhang DWR MSAPP github code`、`"10.37188/OPE.20263401.0167" github EPPM MCIAM`、`"APTNet" "104594" "CamVid"`、`"HFINet" "117375" "ABPRM" EAPPM`、`"ARFNet" Ren Gao HDAPM 117640 github`、`"Three-branch Scale-aware Network" "78.7" TSNet PIDNet EAPPM BGFM`、`"RTSSNet" "29931-1" github Bag`、`RegSeg Cityscapes CamVid dilation ablation`、`DSAFNet MDFA Table 9`、`SDAMNet DCASPP 39205000`。沿论文预览的被引文找到 TSNet，查 Crossref 核对 Elsevier 出版年月，并对照 PIDNet 正文表 3/5/6 与本地代码。无公开全文或完整代码的记录保留证据等级。

- [SDAMNet，Shan Zhao 等，2024-08-16](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359814/)：DeepLabv3+ 的 DCASPP + SFFM + SCSDM 改版；表 8、9 是 Aerial Drone Image 与 UDD6 两套航拍数据，作者 mIoU 提升 +2.89/+2.13 个百分点与 PIDNet、Cityscapes/CamVid 无关。未找到作者代码；仅保留为低优先迁移线索。
- [Discrepancy-Guided Semantic Segmentation with Boundary Detail Enhancement for Traffic Scenes](https://www.mdpi.com/1424-8220/26/9/2738)：消融以 SegFormer-B2 为基线，PIDNet-Small 是横向引用，排除直接改版。
- [HDC/DUC](https://arxiv.org/abs/1702.08502) 与 [DenseASPP](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html) 是一般方法来源；HDC 与 RegSeg 的静态防漏格对照重合，DenseASPP 由本调查另一节的直接 PIDNet 改版覆盖。未把通用方法、仅比较 PIDNet 的网络和同名检测任务计入直接改版。
