# PIDNet 改进方法的检索记录

检索、核表及代码打开日期：**2026-11-11（Asia/Shanghai）**。目标为 Cityscapes / CamVid 单目 RGB，精度优先；先找论文确实改过 PIDNet 的情况，再查三分支近邻和迁移机制。主表含 31 个有一手出处的独立记录（其中 3 个来自补充报告的题名复核），摘要/题录也计入发现但单列证据等级；MEP-Net、PILOT 的不同版本分别各算一篇。方法来源和排除项不计主表。

| 轮次 | 实际查询或追索方式 | 处理结果 |
| --- | --- | --- |
| 直接改造，中英组合 | improved PIDNet / based on PIDNet + semantic segmentation / Cityscapes / CamVid / road；改进 PIDNet 语义分割、三分支、车道、轻量化、蒸馏；PIDNet+、PIDNet++、PIDNetV2、Lite-PIDNet、E-PIDNet。 | 找到 Tan/Jin、RSM-Optimizer、Lian、PIDNet-LW；名称变种不因重名即认作论文。[具体 DOI 及表](direct-cases.md)。 |
| 空洞、池化与结构 | PIDNet + dilated / replacing / ASPP / PAPPM / Bag；FADC PIDNet Cityscapes bottleneck config；EPPM MCIAM；TBMSNet DWR MSAPP；ARFNet HDAPM；APTNet CamVid；RTSSNet Bag；RegSeg dilation ablation。 | CVPR 正文/补充核 FADC；中文期刊核 EPPM；TBMSNet 表 2/5/8 与原 PIDNet 表对读；沿 HFINet 出版页的被引文找到 TSNet。预览、占位代码及引用行单列。[上下文记录](context-cases.md)。 |
| 域外与训练侧 | improved PIDNet + 农业 / 医学 / 工业 / 航拍 / RGB-T；Cherry-Net SwiftRep；改进PIDNet 水位线／玉米倒伏；PILOT replay-free；Agri-PIDNet-S；Pixel-weighted Domain Adaptation APSIPA2025_P572；真实世界超分辨率 语义分割联合 PIDNet；MEP-Net；Boolean Structured Neurons PIDNet；PID-Former coal flow。 | 区分 RGB-T、DSM、超声、合成退化、增量、农业等，原表只能回答域内问题。PILOT 追至正式版；MEP-Net 核 Europe PMC 正文 XML；Agri 追论文代码和结果 CSV；BS-PIDNet 从仓库 README 追 Zenodo 技术稿。 |
| 作者代码与独立补证 | 论文 Data/Code Availability、题名/DOI/作者/模块＋GitHub/Gitee，查 FADC、唐卡、PILOT、Agri、BS-PIDNet、APTNet、RegSeg 等。 | 检查实际代码文件、配置或提交树；唐卡原报告与发布实现差异、FADC 无 PIDNet 配置、APTNet 仅占位。只读代码和元数据，没加载权重。 |
| 补充报告互补 | 题名组合 AWA-PIDNet / piston pins；PIDFormer / BiFormer / 3120274；Key Aircraft Components / Bai；MECS-ECA / MEP-Net；FreqFusion 2408.12879；DySample 2308.15085；Strip Pooling 2003.13328；RDRNet 2406.12496；EMA 权重与 EMA 注意力。 | 交叉核 SPIE/IEEE/Crossref 题录、作者全文表与源码，[互补核查](complementary-review.md) 区分题录存在与原表可核；排除将其他底模或任务数字当 PIDNet 成绩。 |

筛选规则：工作至少要可定位到出版社题录、论文/预印本、原会议论文集或作者代码。标题相近的工作核作者、年份、DOI、任务，避免把 PIDFormer 和 PID-Former、ARFNet 和任等中文论文合并。数值首先看表头、脚注、评估分割、计时协议，再看正文；如正文冲突，保留原表和冲突。来源中带文献编号的 PIDNet 不能自动算同环境重训。没有验证的数值不补全，未发现作者仓库不等于没有源码。只归因同文明确受控比较，作者值可列但不随意作差；所有延迟结论只适用于原条件。

排除：同名行人闯入检测 PIDNet、CT 截断伪影 PIDNET、物理植被含水量 PIDNet；只把 PIDNet 当参考的 SegFormer-B2 衍生、解释性研究、对抗补丁、一般 UAVid 移植、TIA；Optica 专用孔洞任务不进入 RGB 道路比较。SDAMNet 是 DeepLabv3+ 航拍改造，列[附录线索](context-cases.md)；没有依据将其当 PIDNet。WACV 野火蒸馏没有同设无蒸馏 PIDNet 对照，EUSIPCO KD-NAS 的摘要差值对 CNN-I2I，均不用于推荐的 PIDNet 增益。

限制：英文/中文网页、作者提供的公开全文及仓库可查范围，未覆盖所有需授权的数据库与论文；SPIE/IEEE/Springer/SSRN 部分访问受限，Crossref 及 GitHub API 曾限流，改用已公开出版社预览、公开 PDF CDN、原会议目录、仓库网页或 raw 文件，无订阅全文不推断。检索引擎相对日期不作元数据；报告不称穷尽。本轮未更改模型、训练/评估配置、权重、数据或两份原有报告，也未运行完整模型；互补仅执行四项合成 CPU 张量性质检查。

