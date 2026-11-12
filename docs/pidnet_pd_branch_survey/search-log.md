# 检索记录：PIDNet 的 P / D 分支改进方向

> **Agent:** DEEPSEEK-WORKER（opencode-go-chat/deepseek-v4.1-flash）
> **Date:** 2026-11-11
> **Ticket:** T-20261111-001
> **产出:** `report.md`（同目录）

## 1. 本次检索先决条件（必须先读）

本报告的文献层**不是**一次正常的外部检索。本次会话的对外网络只有国内站点可达，学术站点全部不可达。实测记录（`curl -sS -o /dev/null -w "%{http_code}"`，2026-11-11）：

| 站点 | 结果 | 说明 |
| --- | --- | --- |
| arxiv.org / export.arxiv.org | `000` / `502` | TLS 握手失败（`error:00000000:lib(0)::reason(0)`）；DNS 把 arxiv.org 解析到 `2001:2::20` |
| openaccess.thecvf.com | `000` | 同上 |
| www.mdpi.com、link.springer.com、ieeexplore.ieee.org、sciencedirect.com | `000` | 同上 |
| github.com、raw.githubusercontent.com、cdn.jsdelivr.net | `000` | 同上 |
| api.semanticscholar.org、www.semanticscholar.org | `000` | 同上 |
| r.jina.ai、papers.cool | `000` | 同上 |
| 本机 Clash 代理（127.0.0.1:7897） | 进程在、端口在 | 国内站点走代理正常，境外站点全部 TLS 失败；另试 7890/1080/10809 等 11 个常见端口，均无可用出口 |
| cn.bing.com | `200` | 可用，但见下 |
| www.baidu.com、gitee.com、gitcode.com、mirrors.aliyun.com、hub.baai.ac.cn | `200` | 可用 |
| zhuanzhi.ai、papers.cool、aminer 检索页、BAAI 检索页 | `000` / JS 空壳 | 无法取正文 |

必应国内版的行为很特殊：对包含 `PIDNet`、`BiSeNet` 这类已在索引里的专有名词的查询能返回相关结果；对 `STDC`、`PP-LiteSeg`、`CASENet`、`PiDiNet`、`SegFix`、`DDRNet` 以及 `site:` 语法、纯中文通用词组合，全部回落到无关结果（超星登录页、豆包广告、聚丙烯百科、拼多多客服、抖音、QQ 邮箱等）。这不是「没搜到」，而是「搜索引擎返回了别的查询的结果」。

**结论：本轮没有拿到任何一篇新论文的原表。** 报告的文献部分因此分为两类：仓库既有文档里「前轮已核原表」的条目（可直接引用，数字带出处），以及凭领域知识给出的题录/机制条目（标注「未核」，不给数字）。这一限制写进了 `report.md` §0 与 §8。

## 2. 检索式与实际结果（23 条）

全部为必应国内版（`https://cn.bing.com/search?q=...`），逐条执行。结果列只写「相关/无关」，相关的给出唯一文档。

| # | 检索式 | 结果 | 说明 |
| --- | --- | --- | --- |
| 1 | PIDNet boundary branch | 相关 | arxiv 2206.02066、GitHub XuJiacong/PIDNet、知乎 p/626535727、CSDN 135492389、GitHub anhphan2705/im_pidnet、arxiv 2605.08945（同名不同文） |
| 2 | Rethinking BiSeNet For Real-time Semantic Segmentation STDC arxiv | 无关 | 回落 |
| 3 | STDC Rethinking BiSeNet real-time semantic segmentation | 无关 | 回落（拼多多客服结果） |
| 4 | STDC 实时语义分割 细节分支 注意力融合引导 | 无关 | 回落（超星登录） |
| 5 | PP-LiteSeg 轻量级实时语义分割 注意力融合模块 | 无关 | 回落（聚丙烯百科） |
| 6 | PIDNet 边界分支 改进 | 相关 | 同 #1 的 6 个文档 |
| 7 | site:arxiv.org CASENet semantic edge detection | 无关 | 回落 |
| 8 | STDC-Seg | 无关 | 回落（豆包） |
| 9 | PointRend 点渲染 图像分割 论文笔记 | 无关 | 回落（豆包） |
| 10 | BiSeNet 双分支 细节分支 语义分割 解析 | 相关 | 知乎 p/141692672、p/587966908、CSDN 122046087、GitHub CoinCheung/BiSeNet、arxiv 1808.00897 |
| 11 | 语义分割 边界损失 距离变换 | 无关 | 回落（语义学百科） |
| 12 | CASENet 语义边缘检测 | 无关 | 回落 |
| 13 | PiDiNet 像素差网络 边缘检测 | 无关 | 回落（抖音） |
| 14 | SegFix 分割 边界修正 | 无关 | 回落（科研测试平台） |
| 15 | PIDNet D分支 边界 改进 | 相关 | 同 #1 |
| 16 | PIDNet 改进 论文 2025 语义分割 | 相关 | 同 #1（索引里只有这些 PIDNet 页面） |
| 17 | DDRNet 深度双分辨率网络 语义分割 | 无关 | 回落（QQ 邮箱） |
| 18 | PIDNet 边界分支 注意力 Bag 改进 | 相关 | 同 #1 |
| 19 | PIDNet 高分辨率 细节分支 改进 | 相关 | 同 #1 |
| 20 | improved PIDNet GitHub 改进 仓库 | 无关 | 回落（词典页） |
| 21 | arxiv 1812.07032 boundary loss segmentation | 无关 | 只返回 arXiv 首页类结果，无该文 |
| 22 | arxiv 2108.10772 pixel difference networks edge detection | 无关 | 同上 |
| 23 | arxiv 1703.09859 CASENet category-aware semantic edge | 无关 | 同上 |

**统计：** 23 条检索式，7 条返回相关结果（全部落在同一批 PIDNet/BiSeNet 中文页面），16 条回落；去重后唯一相关文档 13 个（2 个 arXiv 摘要页、4 个 GitHub 仓库页、5 篇中文解析、1 篇 BiSeNet 原论文、1 个同名不同文的 PIDNet）。其中**没有一篇**给出 P 或 D 分支的改进消融数字，超出 `docs/pidnet_methods_survey/` 已有内容的信息量为零。

## 3. 筛选标准

纳入需同时满足：

1. 改动落在 P（细节/高分辨率流）、D（边界流）或两者交互上，或提供 P/D 的设计原则；
2. 有可定位的题录（出版社页、arXiv、会议论文集）或可打开的仓库；
3. 若是「论文报告数字」，必须能说清对照协议；说不清就只作方向线索。

排除规则：

- 只改 I 分支上下文的工作（DenseASPP/PMPM/FADC/EPPM/MSAPP/LCPPM 等）：前轮已覆盖，本报告只在计数时引用。
- 同名不同文：arXiv 2605.08945 的 *PIDNet: Progressive Implicit Decouple Network*（多模态分割），与 CVPR 2023 的 PIDNet 无关，排除。
- 只有 README、没有可核对 diff 的仓库：GitHub `anhphan2705/im_pidnet`（标题称 Improved PIDNet，本轮只能看到仓库描述，没有可核对的文件级改动），排除。
- 中文解析文章（知乎/CSDN 的 PIDNet 结构解读）：只能确认论文机制，不含 P/D 改进实验，作为背景不计数。
- 域外工作（农业/医学/工业/RGB-T/航拍/增量学习）：可作机制线索，不进入街景迁移的收益判断；`report.md` 里逐条标注域外。

## 4. 纳入结果

- **外部文献/项目 40 项**：与 PIDNet 直接相关 17 项（R1–R17），通用机制 23 项（G1–G23）。另有 DySample、FreqFusion 两项前轮已覆盖，只在正文提及、不单独编号。
- 其中**本轮能一手确认题录的**：R1（arXiv 页面在索引中，题名/摘要片段可见）；R2–R17 的题录来自仓库既有文档或并行报告（后者本轮有完整外网，见第 7 节）。
- 其中**已核原表数字的**：R1–R6、R9–R11、R14–R16（12 项，前轮）；R17、G12、G15、G16、G22、G23（并行报告或前轮）；G19（并行报告核了表 3）。
- 其余 17 项（G1–G11、G13、G14、G17、G18、G20、G21）为「未核」：机制描述来自领域知识，arXiv 编号未核，不给数字。

## 5. 与前轮检索记录的关系

- `docs/pidnet_methods_survey/search-log.md`：31 项，聚焦「谁改了 PIDNet」。本轮从其中抽出 16 项与 P/D 相关的，重排为「改 P 的 / 改 D 的 / 改融合的」，并补上结构层面的解读；没有新增直接改 PIDNet 的论文（本轮网络条件下也做不到）。
- `docs/pidnet_accuracy_research/literature.md`：边界与小目标的一手文献（Boundary IoU、Gated-SCNN、SegFix、ABL、Lovász）。本轮把其中 5 项纳入 D 分支的监督/修正方向，并新增「D 的结构与门控」这一层，两者不重复。
- `docs/pidnet_pd_research/plan.md`（Sol Lead，另一 ticket）：同主题的并行任务。我起草时该目录只有 plan；写完后它产出 `p-evidence.md`、`d-evidence.md`、`fusion-evidence.md`、`notes.md`、`report.md`。我用它交叉核对了四处，见下节。

## 6. 交叉核对（并行报告已核、我本轮核不了的部分）

并行任务（`docs/pidnet_pd_research/`）在同期有完整外网，能打开 arXiv PDF 与 GitHub 文件。我只在写完之后用它校对了以下四处，并在正文对应位置注明：

| 我原来的写法 | 核对结果 | 处理 |
| --- | --- | --- |
| GSCNN「GCL 使 mIoU 72.7→74.3、5px F1 69.8→73.3」 | 72.7→74.3 是 shape stream + dual-task 整块；表 4 单独 GCL 的 F@5 是 69.8→70.8，加 dual-task 才是 73.0 | 改写为分列，D1/P3/G12 三处同步 |
| ABL「val 79.5→80.5」 | CE 79.5→单加 ABL 79.6（+0.1）；CE+Lovász 80.2→80.5（+0.3） | 改写为两行，D5/G15 同步 |
| PiDiNet 编号 2108.10772 | 实际为 arXiv 2108.07009，BSDS500 表 3 ODS 0.736→0.757 | 编号与数字更正，G19/D6/D8 同步 |
| 未收录 BEVANet | ICIP 2025，作者代码 MIT，P 侧 SDLSKA 块，固定 BA/Bag 的 Cityscapes 表 4 77.77→78.60→78.86 | 新增 R17，扩写 P6，§7.4 记录分歧 |

没有采信的部分：并行报告的 12 项候选表里，TSNet/HFINet 与我一样只到摘要级；PILOT、FreqFusion、DySample 的结论与前轮文档一致，不重复引入。

## 7. 限制与补核清单

1. **没有原表数字的新文献。** 若要把 G1–G11、G13、G14、G17–G21 里的任何一项写进论文，必须先打开原文核三件事：对照是否为同一骨干/同一 split、是否报告了延迟、是否有消融。这三件事靠摘要和二手笔记都判断不了。
2. **arXiv 编号未核。** 18 条未核条目的编号是凭记忆写的，可能与正式版编号不同（尤其 STDC-Seg、PP-LiteSeg、CascadePSP 这几篇有过版本变动）。用题名检索比用编号稳。
3. **仓库路径未核。** 除 R6、R9、R14 的作者代码路径（前轮已打开）外，正文提到的通用仓库名（STDC-Seg、CoinCheung/BiSeNet、DeepGuidedFilter、CascadePSP、openseg.pytorch）本轮都无法访问 GitHub 确认。
4. **本机计数只到 S/M。** L 只做了参数计数（37.314M），没有做乘加计数（CPU 前向太慢，4GB 卡也不适合跑 2048×1024 的 L）。L 的计算占比结论未验证。
5. **没有真实数据。** 所有「边界像素占比」「边界带宽敏感性」都是经验判断，未在本仓库的数据上统计（数据软链当前不可用，见 `docs/pidnet_accuracy_research/report.md`）。
