# 交接包恢复与外部输入

1. 用有权限的GitHub账号下载同一Release的ZIP和`SHA256SUMS.txt`，校验整包后解压。ZIP内容以仓库根目录为基准，包含`af-research/`；不要再多套一层`af-research`。
2. `MANIFEST.json`给出每个材料的相对路径、大小和SHA-256。旧报告引用的源码须优先对照同目录`source-snapshot/`，不要拿新版本覆盖历史快照。
3. 原始照片不在包里。`af-research/private-data/recognition-v1`与`recognition-v2`包含原manifest、来源记录及官方CSV元数据。按manifest的ID、URL／来源和SHA恢复照片；保持已有train/validation/test划分，不能重新随机划分。只复算保存预测的指标通常不需要照片，但图片QA及新的前向检查需要原图。
4. 完整原始S5组件另行提供。`research_paths.py`支持`--source /path/to/components`或`S5_AF_FIRMWARE_ROOT`。模型文件及SHA见子项目README；ARM转换器回放额外需要原`03_program.bin`和`verification.json`。
5. 外部Paddle模型和既有输出保留；vendor只带原LICENSE及两个适配器所需预处理文件。需要完整PaddleDetection时，恢复报告固定的commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5`，并参照`af-research/patches/`及外部基线说明。`verify_picodet_adapter.py`还要读取第一张train图，不能在照片未恢复时声称完全离线可跑。
6. 参考论文／手册PDF另按`artifacts/recognition-ranking-repair/box-contract/report.json`中`sources`的URL下载，并核对已有SHA。原报告不会因未携带PDF而被改写。
7. QEMU转换器回放使用显式`--firmware-root`和`--qemu`，另需`/usr/bin/clang`。历史`firmware-contract/*.py`及原始负对照快照保留开发机绝对路径；如需换机重跑，请创建参数化副本，保留旧文件及SHA。

实际验证平台为macOS arm64。原生Windows的`resource`依赖、Linux/WSL、CEVA运行时及机内执行没有获得本次验证。程序能导入、损失梯度正确、QEMU边界函数回放通过，均不等于模型检测验收通过。

当前停止在排序候选和脸／头语义核对阶段。不要自动运行旧`train_recognition.py` CLI：它仍面向早期布局筛选。所有新训练应先阅读交接说明、明确目标语义并解决现有误报／小脸门槛；附件里的checkpoint不是可刷固件。
