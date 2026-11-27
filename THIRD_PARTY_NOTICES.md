# 第三方材料与外部输入

- Open Images：标注与来源元数据来自官方数据集；标注许可及图像各自的作者／许可在 `af-research/private-data/*/provenance.json` 和 manifest 中保留（位于Release附件）。原始照片不随本交接包分发。数据入口见 `af-research/docs/RECOGNITION_DATA.md`。
- PaddleDetection／PicoDet／BlazeFace：用于外部对照，来源版本、下载地址和SHA保存在 `af-research/external_baseline_models.py`、外部基线报告及附件的模型目录中。第三方项目源码缓存不随包分发；其Apache-2.0许可文件单独保留。
- Panasonic DC-S5固件与文档：完整固件及官方手册PDF不随包提供。研究报告保留来源哈希、少量原ARM转换器指令片段和分析结论。实验checkpoint包含从用户提供组件恢复的冻结骨干和实验参数，不是可刷固件或已通过原机验证的升级。
- QEMU／编译器／Python依赖：仅记录验证版本、命令和来源哈希；二进制、虚拟环境及工具链缓存不随包分发。

本文件用于交接来源信息，不替代各材料原有许可。历史研究源码快照和报告保持原文，包括原开发机路径。
