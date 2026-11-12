# 直接改造 PIDNet：原文证据摘录

检索与核证日期：2026-11-11（北京时间）。优先核对作者论文全文、表格和发布者元数据；仅有作者摘要、二次索引或会议题录的项目标注证据限度。表内成对列值不自动视为同协议；百分点差只引用明确受控的同文消融或作者明确对照。以下不涉及模型运行。

## 城市道路 RGB：直接修改，优先参考

### Tan / Jin：DenseASPP、HRNet 式提取、Bag 交叉注意力

Guangxing Tan、Yangying Jin，2025-02-23，*A Semantic Segmentation Method for Road Sensing Images Based on an Improved PIDNet Model*，[Electronics 14(5):871 正文](https://www.mdpi.com/2079-9292/14/5/871)，[出版社 PDF](https://mdpi-res.com/d_attachment/electronics/electronics-14-00871/article_deploy/electronics-14-00871.pdf)。直接改 PIDNet：共享特征提取引入 HRNet 式多分辨率交互，I 分支用 DenseASPP 替代原金字塔上下文模块，Bag 用交叉注意力融合 P/I/D。§4.2 与图 3 描述四个空洞卷积尺度及残差映射；源码缺席时优先按模块接口验证，不能从公式反推确定实现。

| 原表与数据 | PIDNet-S | 改进版 | 条件与核证限度 |
| --- | ---: | ---: | --- |
| 表 1 Cityscapes val | 78.8% | 81.0% | 表内分开列 val/test；基线标有原 PIDNet 引文 |
| 表 1 Cityscapes test | 78.6% | 81.1% | 101 / 99 FPS，RTX 3090，2048×1024；是否同训练与推理协议未完全交代 |
| 表 2 CamVid | 80.1% | 82.6% | 160 / 155 FPS，RTX 3090；未核到该表的评估 split、输入尺寸、预训练及 TTA |
| 表 3 Cityscapes 消融 | 78.6% | DenseASPP 79.0%；HRNet 79.1%；两者 80.5%；三者 81.1% | DenseASPP 单项 +0.4 pp；三项 +2.5 pp；评估 split 未标明；无 cross-only 行 |

§5.1 写 Cityscapes 19 类，2975/500/1525；§5.2 写 ImageNet 预训练、620 epoch、batch 6、1024×1024 训练、随机缩放 0.5–2，未明确评估 TTA 或计时 batch/精度/端到端边界。表 3 PDF（p.11）逐格核对：HRNet+Cross Attention 为 **79.9%**，§5.4 正文称 **79.7%**；DenseASPP+Cross 为 79.5%。§5.3 表 2 的 CamVid 82.6%/155 FPS 后一段误称 CityScapes；正文“2.3%”也与表 1 内两组差值不吻合。表 4 标称 val 却只有 16 类且含 Lane marking，不能据此重算[标准 19 类](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions) mIoU。数据声明指向正文；题名、作者和模块名检索未找到可核归属的完整实现。

项目价值：DenseASPP 是改动面最小且有街景单项消融的候选；三项 81.1% 不能全归因于 DenseASPP。按相同 RGB、尺寸、初始化、验证集与 TTA 策略重跑单项，再决定是否引入高分辨率提取和融合注意力。

### RSM-Optimizer：RF-FEM、PMPM、SUFM

Xiaohong Zhang、Wenwen Zong、Yaning Jiang，2025-03-11，*RSM-Optimizer: Branch Optimization for Dual- or Multi-Branch Semantic Segmentation Networks*，[Electronics 14(6):1109 正文](https://www.mdpi.com/2079-9292/14/6/1109)。通用分支优化方法，作者确实接入 PIDNet-S（还接入 BiSeNetV2 和 DDRNet-23-slim），故这里是直接修改证据。RF-FEM 扩大 P 细节分支感受野；PMPM 把 I 语义分支 PAPPM 的平均池化换成方形/横条/竖条混合池化；SUFM 用两级加权特征融合和 CARAFE，从 1/64 到 1/32 到 1/16 后再上采样对齐 P 分支（§3、图 1–4）。

| Cityscapes 表 1 | mIoU | FPS | 参数 M | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| PIDNet-S | 78.3% | 125 | 7.62 | 47.6 |
| RF-FEM | 78.8% | 120 | 7.92 | 57.2 |
| PMPM | 79.1% | 103 | 8.41 | 47.9 |
| RF-FEM + PMPM | 79.4% | 98 | 8.71 | 57.7 |
| RF-FEM + PMPM + SUFM | 79.8% | 94 | 9.92 | 69.3 |

表 1 显示 RF-FEM 单项 +0.5 pp、PMPM +0.8 pp、全套 +1.5 pp；没有 SUFM 独立行，79.4→79.8 只表示叠加在前两项上。表 2 Cityscapes 重列基线和全套（2048×1024）；表 3 CamVid 列 PIDNet-S 80.1%/153 FPS、PIDNet-S* 82.8%/150 FPS（720×960，正文以这两行比较）。§4.1 写 Cityscapes fine 19 类 2975/500/1525、CamVid 11 类 367/101/233，500/200 epoch、batch 6、SGD、5 张 RTX 4090 的统一训练服务器。表 1–3 未标分数来自 val 或 test；未见预训练来源、测试增强、精度、计时 GPU 数与批次。§4.3.1 称 PIDNet-S* 为最高，但表 2 PIDNet-M 80.1%，高于 79.8%；对全表排序采用表值。数据声明只给 Cityscapes 链接；题名、作者和模块词检索未找到作者代码。

项目价值：PMPM 与 DenseASPP 同为 I 末端的一处改造，适合分别对同一 PIDNet-S 做 A/B；表 1 GFLOPs 近似但 125→103 FPS，不应以算力替代延迟测量。全套适合精度优先且可接受减速的后续比较。

### Lian 等：SOM、PAPPM 坐标聚合注意力、PSA

Jihong Lian、Nan Zhang、Yunhong Li、Weige Xue，2024-06-28 至 30（EEI 会议日期），*An Road Scene Semantic Segmentation Algorithm Based on Modified PIDNet*，pp. 1082–1089，[IEEE 正版入口](https://ieeexplore.ieee.org/document/10696235)，[出版社提交的 Crossref 元数据](https://api.crossref.org/works/10.1109%2FEEI63073.2024.10696235)。题录和作者由 Crossref 与[会议论文集目录](https://www.proceedings.com/content/076/076615webtoc.pdf)交叉核实。公开的[原摘要复载](https://backend.eurekamag.com/research/095/910/095910006.php)描述 SOM 在连续下采样中对关键信息加权、PAPPM 内嵌坐标聚合注意力、另外使用 polarized self-attention（PSA）；具体 PSA 插入层未核。摘要称 Cityscapes **83.52% mIoU**。IEEE Xplore 被浏览器验证拦住，未取得可核正文/原表、同文基线、变体、split、分辨率、预训练、TTA 或计时；题名/DOI/作者检索未找到可核作者仓库。仅作三个设计线索，不把 83.52 排进街景对照或归为单模块效果。

## 城市道路辅助验证与域外迁移

### Wu 等：ECA-Pag、LGFM、PConv3

Jiao Wu、Tiejun Wang、Xiaoyan Hu、LingMei Tao、Tianjiao Duan、Yanjiao Wei，2026-05-14，*Research on the semantic segmentation of Thangka images via an improved PIDNet*，[PLOS One 21(5):e0332164](https://doi.org/10.1371/journal.pone.0332164)。直接修改三分支：P 的 Pag 加 ECA 和残差，P 末端 LGFM 融合 CoT 局部特征及全局注意力，D 分支加入 PConv3（§2.1–2.4）。

- **Cityscapes**：§3.3 / [表 8](https://pmc.ncbi.nlm.nih.gov/articles/PMC13175359/#pone.0332164.t008) 为 PIDNet 74.38%、改进模型 75.40%，两行均 RTX 3090、1024×1024 crop、80k；512×512/160k 是表内 Swin-B 的条件，正文的“可比尺寸”表述不清。表注称共用 80k 调度；具体变体、评估 split、预训练、TTA 和 Cityscapes 计时未核定。唐卡的 109 FPS 与该表无关。
- **唐卡（单独域）**：自建 783 张原图，12 个标注类别、512×512；§2.6 先描述扩至 4698，再划分 3776/922，未证实按原图分组防止近重复穿越；§2.5 写从 3776 留 755 验证。§2.8 写 RTX 3090 实验环境（§2.5）、Adam、45k 迭代上限、5 种随机种子。§3.2 / [表 5](https://doi.org/10.1371/journal.pone.0332164.t005)：同训练参数下 PIDNet-M 70.20±0.17% mIoU / 35.97±0.13% 边界 F 分数 / 104.63 FPS；ECA-Pag 单项 70.76%，PConv3 72.00%，LGFM 74.14%/85.98 FPS；全套 73.33±0.12% / 40.06±0.17% / 109.06 FPS。全套边界分数最好，单 LGFM 的 mIoU 更高。摘要给 73.28/40.01/109.03，采用表 2/5 的重复均值；唐卡预训练来源、TTA 未明确。
- **公开材料**：§2.8 明链作者库 [JikD-591/thangka-seg](https://github.com/JikD-591/thangka-seg)，[Zenodo 19498531](https://zenodo.org/records/19498531) 版本元数据确认发布者 Wu, Jiao，列图像/掩码/JSON 和 `best_mIoU_iter_48000.pth`（未下载）。[`newpidnet.py`](https://github.com/JikD-591/thangka-seg/blob/main/mmseg/models/backbones/newpidnet.py#L112) 有 ECA/残差 Pag、P 末端 LGFM、D 分支部分卷积；实际 `Partial_conv3` 对部分通道执行卷积，未见 §2.4 所述可学习空间掩码。仓库的 [`configs/pidnet`](https://github.com/JikD-591/thangka-seg/tree/main/configs/pidnet) Cityscapes S/M/L 入口是普通 PIDNet 120k/1024 crop，不能视为改进版 80k 表 8 配置。

项目价值：ECA-Pag 是可读源码的低侵入方案；LGFM 在高分辨率 P 分支上的延迟及显存须实测；暂不能用域内结果估计街景增益。

### Zheng 等：UAVid 与 Caihua Village 的 PAPPM / head

Zezhong Zheng、Zixuan Teng、Shuang Yu、Qiang Liu，2026-02-19 上载，*Semantic Segmentation of UAV Imagery in Urban and Rural Scenes Based on Improved PIDNet*，[SSRN 作者预印本入口](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6271048)，[作者提交的 Crossref 摘要/元数据](https://api.crossref.org/works/10.2139/ssrn.6271048)。直接重做 I 分支 PAPPM 与轻量化语义头。摘要的两组对照为 UAVid 70.73→71.30%，Caihua Village 87.27→89.17%；改头后摘要分别给 72.33%、89.92%，称参数和 FLOPs 最大降幅 88%；未取得原表，不能判断 88% 是否仅语义头、模型变体、两级变更是否严格递增同设。SSRN PDF 请求返回 403；split、预训练、TTA、输入、计时及源码未核。题名/作者/Crossref 关系未找到期刊终版；[IGARSS 2026 同方向海报](https://2026.ieeeigarss.org/view_paper.php?PaperNum=4215&SessionID=1388) 作者组合不同，不可冒认为终版。RGB 航拍光学域可提供上下文与 head 设计线索，不能转算街景增益。

### Li 等：DGHF-PIDNet（RGB-T）

Haiyu Li、Tianyu Liu、Shilong Wu、Zihao Song、Haoran Dong、Jinfeng Du、Xiangyu Wang，2026-05-31，*Real-time RGB-T semantic segmentation via hierarchical semantic guidance and detail-preserving PIDNet*，[J. Real-Time Image Processing 23:108](https://link.springer.com/article/10.1007/s11554-026-01903-2)。保留 P/I/D，P 加 DAGRConv（训练时方向卷积及输入相关门控、作者称推理分支折叠为单个 3×3），用 HF-BIGate 替换 P/I 交互，逐级把压缩语义送入细节分支。出版社摘要称 SemanticRT、FMB、PST900 均优于 PIDNet，7.85M、218 FPS；全文需订阅，未核原表、绝对精度、基线变体、split、输入/预训练/TTA/计时硬件，不能推断完整动态门控如何折叠。出版社称代码可合理请求作者提供；以标题和模块名未找到公开作者仓库。只考虑 RGB 条件下独立测试门控，不引用 RGB-T 数值为单目收益。

### Cui 等：Cherry-Net，樱桃成熟度分割

Jie Cui、Lilian Zhang、Lutao Gao、Chunhui Bai、Linnan Yang，2025-09-03，*Cherry-Net: real-time segmentation algorithm of cherry maturity based on improved PIDNet*，[Frontiers in Plant Science 16:1607205 全文](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1607205/full)。属于从 PIDNet 大幅改造的域外衍生版：换 SwiftFormer-XS backbone、移除若干原损失/残差组件，SwiftRep-Hybrid 用于 backbone 的局部/全局融合，LFE 用于末端特征融合；§3.1 把 LFE 原替换点分别称 Bag 与 pixel attention guidance，复现应按图 3 与 §2.3 核实。

Cherry CO Dataset-ripeness 共 3006 张，train/val/test 1804/601/601；原图 1328×1328、训练随机裁剪 1312×1312，4090D，SGD、400 epoch、batch 6。表 4 消融：PIDNet-S 70.19% mIoU，换 backbone 并去冗余损失 71.64%，再加 SwiftRep 72.00%，再加 LFE 72.20%；表 6 并列为 70.19→72.20%、100.02→104.63 FPS、7.6→5.9M。表 4/6 表注明确 val，§2.7 写 test；保留该冲突。表 7 在同设 Jetson TX2 / TensorRT FP16 / 1312×1312 上为 10.3→10.8 FPS、97→93 ms。表 5 三损失同开为 71.20%，§3.1 却称 72.20%；没有依据把组合损失宣传成单独增益。预训练、TTA、GPU 计时边界未见明确，论文数据声明指向正文/补充材料，题名/作者/SwiftRep 检索未找到可核作者仓库。RGB 域外的 LFE 适合做独立融合消融，整套 backbone 替换成本高，且没有街景验证。

### 李仲等：LPEM / LCPPM，河流水位线

李仲、冒睿瑞、王晓龙、王根一、安国成，2024 年 8 月刊（期刊标注 2024-01-16 发布），《基于改进PIDNet的水位线检测算法》，[《计算机工程》50(8):102–112](https://www.ecice06.com/EN/Y2024/V50/I8/102)，[期刊 PDF](https://www.ecice06.com/CN/article/downloadArticleFile.do?attachType=PDF&id=32497)。直接改 PIDNet-S/M：I 分支下采样引入带通道注意力的 LPEM；上下文池化换成更少并行层级和通道加权的 LCPPM。

PDF p.108 表 3/4：基线 S/M mIoU 87.96%/90.31%，LPEM 单项 88.20%/90.61%，联合 LCPPM 89.04%/91.35%。基线 M 到联合版表内推理为 38.9→38.6 ms，§5.2 同述 −0.3 ms；摘要 −0.9 ms 与此不符。§4 描述 3523 张五场景河流图像、train/val/test 9:0.5:0.5（同段“各 704”相乘为 3520，保留差异）；960×720 输入，表 1 标 NVIDIA Tesla A30，单图平均推理；mIoU 仅在标注水线纵向上下各 60 像素邻域计算。未核到预训练、TTA 或作者仓库。该指标不可和街景全图 mIoU 比较；LPEM 或 LCPPM 可作为 I 分支低成本消融线索。

### 李嘉豪等：EMA-PIDNet，玉米倒伏（RGB+DSM）

李嘉豪、刘开东、柴子凯、宁纪峰、杨蜀秦，2025 年 11 月，《基于数字表面模型和改进PIDNet的玉米倒伏无人机遥感监测研究》，[《农业机械学报》56(11):378–386](https://www.j-csam.org/jcsam/article/abstract/20251136)，[期刊 PDF](https://nyjxxb.ijournals.cn/jcsam/article/pdf/20251136)。修改为四通道输入（可见光+DSM），在 I 主干及 I→P、I→D 融合前各接一个 EMA（图 4、§2.2）。按地块拆分训练/验证/测试为地块 1–2 / 3 / 4；每生育期 9513 训练图块（含扩充）、1073 验证、960 测试，512×512。实验机器 2×RTX 3090，SGD、batch 16；未见 PIDNet-S/M/L 标识、预训练、TTA、FPS 或作者代码。

原表 2–3（p.384），mIoU 按吐丝/乳熟/腊熟顺序：RGB PIDNet **74.09/72.84/72.61%**；加 DSM **79.61/78.70/76.87%**；DSM+I 主干单个 EMA **78.30/78.55/75.68%**；DSM+两处跨分支 EMA **79.58/79.08/76.59%**；DSM+三处 EMA **83.94/82.77/82.64%**。只加 EMA 一处表现下降，两处结果不稳定，完整三处在固定 DSM 条件下领先；无 RGB-only EMA 行。应把传感模态与网络改动分开，不能引用全套增益证明 RGB 街景只加 EMA 有效。

## 已执行检索与排除

- 首轮英文关键词：`10.3390/electronics14050871 PIDNet Tan Jin DenseASPP CamVid Cityscapes`、`10.1109/EEI63073.2024.10696235 PIDNet Lian segmentation`、`10.1371/journal.pone.0332164 PIDNet Thangka segmentation cityscapes`、`PIDNet 6271048 UAV segmentation improved PIDNet journal`。依次追期刊表、MDPI 静态 PDF、IEEE Xplore/Crossref、PLOS 的 PMC 原表及论文作者库、Zenodo API 元数据。
- 扩展关键词：`RSM-Optimizer PIDNet Cityscapes CamVid`、`DGHF-PIDNet github DAGRConv`、`Cherry-Net Jie Cui github SwiftRep`、`改进PIDNet 医学 语义分割 工业 缺陷`、`基于改进PIDNet的水位线检测算法 DOI 李仲 代码 github`、`基于数字表面模型和改进PIDNet EMA GitHub 李嘉豪`。从中文期刊目录定位玉米倒伏的独立记录和 PDF。棉田 Agri-PIDNet-S 由并行条目负责，不重复计入。
- SSRN 用题名、作者和 Crossref bibliographic 查询追版本，未发现期刊终版；同课题方向的 Ba 等 IGARSS 2026 海报不是同一作品。排除把 PIDNet 仅作基线的 [SCPD-DeepLabv3+](https://cjournal.hep.com.cn/1008-1534/CN/1217459861044335093)、[SSFNet](https://www.zjujournals.com/eng/EN/abstract/abstract46942.shtml)，以及仅公开 PIDNet 三变体训练权重的 [UAVid 移植仓库](https://github.com/markopetrov1/real-time-segmentation-pidnet-paper)。Optica 10.1364/AO.572168 的任务域偏离 RGB 道路，未纳入比较。
