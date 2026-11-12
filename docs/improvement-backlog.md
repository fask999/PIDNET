# PIDNet 待改进清单

> **Agent:** Codex /root — 根据用户决定记录待办
> **Date:** 2026-11-11（Asia/Shanghai）

当前以 **PIDNet-L 的 I 分支增强**为实验目标。以下三项先保留为待办，暂不实施；具体方案与消融安排在后续启动时再确认。

I 分支末端支持 PMPM / DenseASPP，内部指定块支持 fixed / FADC。参见 [PIDNet-L 消融用法](pidnet-l-ablation.md)。L 默认 `ABLATION.CONTEXT=auto`（DAPPM）和 `ABLATION.I_ATROUS.MODE=none`，保持 Bag 融合。PAPPM 是显式对照；I 分支组合预设沿用相同边界监督，边界改进另有独立预设。

| 优先级 | 待改进方向 | 首选方案与接入位置 | 状态 |
| --- | --- | --- | --- |
| 1 | P 分支细节增强 | BEVANet 式单个 SDLSKA 空间细节块，放在 `layer4_`、第二次 Pag 前；RF-FEM 作为备选路线。 | 待改进，暂缓 |
| 2 | D 分支语义门控 | 借鉴 GSCNN，用同级 I 特征引导 D；放在 `diff4` 加和后、保存 `temp_d` 前，同时影响辅助边界头与后续融合。 | 待改进，暂缓 |
| 3 | P/I 上采样对齐 | 先确认存在空间错位，再对 `pag3` 的一次上采样尝试 DySample；FreqFusion 作为后续融合候选。 | 待改进，需先验证错位 |

## 后续实施约束

- 三项分别保留独立开关，默认关闭，核对关闭时的默认兼容性。
- 固定届时选定的 I 分支、边界监督、初始化、训练配方和评估协议，避免混入其他变量。
- P、D 先比较基线、仅 P、仅 D、P+D 四组；确认效果后再考虑上采样对齐。
- P 重点观察细小类别 IoU 与轮廓连续性，同时记录高分辨率计算开销。
- D 同时观察整体 mIoU、去 void 的边界指标和细目标误抑制；仅修改 `layer5_d` 输出不会直接改变辅助边界头的输入。
- DySample 需核对共享采样坐标、初始双线性等价性和非整数倍率尺寸；尚未确认错位时不默认启用。
- 使用独立的 train/val 划分选型；CamVid 原配置中的 `trainval` 包含验证集，不能用于验证集消融选型。方案固定后再使用 test。
- 论文收益不等于本仓库收益；当前未实施这三项，也未验证其精度或速度。

## 依据

- [P/D 分支调研与优先级](pidnet_pd_research/report.md)
- [P 分支证据](pidnet_pd_research/p-evidence.md)
- [D 分支证据](pidnet_pd_research/d-evidence.md)
- [融合证据](pidnet_pd_research/fusion-evidence.md)

## 记录

- 2026-11-11，Codex /root：按用户要求，将以上三项列为待改进；当前工作优先转向 I 分支。
- 2026-11-11，Codex /root：核对当前代码，明确 I 分支增强已经实现，区分功能存在与默认启用状态。
- 2026-11-12，sol-lead：明确以 PIDNet-L 为目标，并补充 L 消融用法；三个待办未实施。
