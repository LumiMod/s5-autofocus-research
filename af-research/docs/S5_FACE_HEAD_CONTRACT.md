# S5 人脸／头部框定义核查

核查日期：2026-11-27。只检查现有 train/validation 标注、已保存诊断和官方资料；未读取 test 图像、未运行新推理或优化器。本说明保留 S5 原网络结构，训练仍暂停。

目前可确认：**Open Images 的 Human face 和 Human head 是不同类别；其通用框规范是目标可见范围。原代 DC-S5 官方资料明确声称分别识别头、脸、眼和身体，但没有公开这些类别与本项目恢复网络输出通道的对应关系。** 因此，将当前 `person_face` 输出直接等同于原机的“脸框”仍是假设。

## 1. Open Images 的已确认定义

本地 `private-data/recognition-v2/raw/classes.csv` 与 `hierarchy.json` 区分：

| 类别 | 官方 MID | 当前项目处理 |
|---|---|---|
| Person | `/m/01g317` | Person 及 Man/Woman/Boy/Girl 的 Subcategory 合并为 `person` |
| Human face | `/m/0dzct` | 映射为 `person_face` |
| Human head | `/m/04hgtk` | 没有纳入当前训练 manifest |
| Human body | `/m/02p0tk3` | 没有因 Part 关系并入 Person |

层级中 face 与 head 都出现在 Human body 的 `Part` 下，也有各自的根层条目；它们不是 Person 的 `Subcategory`。`recognition_data.person_subclasses()` 只收集 Person 的 Subcategory，不把 Part 当成同义类。官方 [类别表](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv) 和 [层级 JSON](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json) 是核查来源。

作者论文 §2.4.1（PDF 第 6 页）的通用要求是：

> “the smallest possible box that contains all visible parts of the object”

即包住目标所有可见部分的最小框，**不是 amodal 框**（猜测并补全被遮挡后的整个物体）。这一规范同时适用于 Human face 与 Human head；“head”也不自动意味着补全不可见的头部。作者说明类别细节会有歧义，并向标注者提供类别示例，但本次没有取得 Human face 关于头发、耳朵、额头边界等细节的专用示例，不能据通用规范进一步断言这些部位一概纳入或排除。[作者论文](https://arxiv.org/html/1811.00982v2)

本项目的 bbox 文件实际来自官方 V5 validation，类别表来自 V7；这点由下载地址和文件哈希记录，不能统称为“新采集的 V7 框”。官方 V7 说明仍指定上述论文作为 V5–V7 的数据方法引用。[V7 官方说明](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)

`IsOccluded=1` 表示该目标被另一物体遮挡；它不是“框已补全”的标记。`Confidence` 是恒为 1 的占位字段，不是标注质量评分。官方声明对已有正图像级标签尽量穷尽框，但未标的类别不等于确认不存在。[字段说明](https://storage.googleapis.com/openimages/web/download_v7.html)

## 2. 本地转换与可用的 head 对照

现有转换保留选中 person/face 的原坐标和属性，按原图宽高把归一化坐标还原为连续 xyxy；不额外加一、不扩大脸框、不将脸替换成头。之前 train320 对照确认 883 条目标标注完整保留。

独立导出 `artifacts/recognition-ranking-repair/box-contract/head-reference.json` 使用**同一批原图 ID 的原始 Human head 行**，不修改旧 manifest 或历史指标：

| 项目 split | 所有图片 | 有 head 框图片 | 原始 head 框 | head 含 Depiction 的图片 | 排除这些整图后图片／框 |
|---|---:|---:|---:|---:|---:|
| train | 320 | 261 | 342 | 3 | 258 / 336 |
| validation | 80 | 68 | 97 | 4 | 64 / 93 |

这些行没有 GroupOf，但有 3 / 4 个 Depiction 框。原先的整图过滤只检查 person/face，因此不能认为新增 head 行已经过滤。对照应排除含 head GroupOf/Depiction 的整幅图片，而不是丢掉一个框后把剩余像素当背景。没有 head 框的图片不视为 head 负图。属性、CSV 行号、原坐标、文件哈希、split 全部保留。来源标注的许可为 Google LLC CC BY 4.0，原照片许可另行保留在既有 manifest。[许可说明](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)

官方 CSV 没有把某 face 与某 head 绑定到同一人物实例的 ID。这里只能报告同图几何关系；不能将最大 IoU 自动解释为已证实的人物对应。

## 3. 原代 DC-S5 官方资料能证明什么

核对的是 **DC-S5** 手册 `DVQP2197ZD`，不是 DC-S5M2/S5II。手册第 102–103 页的显示与选择规则为：

| 对象 | 官方资料能确认的行为 | 不能据此推出 |
|---|---|---|
| 脸 | 检出脸时显示 AF 区域；最多 15 张脸 | DSP 原始 bbox 的边界规范 |
| 眼 | 在黄色目标框内检眼，优先对焦较靠近相机的一眼；可切换目标 | 当前两个检测头内直接包含眼坐标 |
| 身体 | 检出全身或上半身；人体和动物身体合计最多 3 个 | Open Images Person 与原机 body 的标签完全一致 |
| 头 | 手册该节没有提供头部专属显示框定义；2020 年原代 S5 发布稿明确说明头与眼、脸、身体分别识别，并用于转身等情况的持续跟踪 | `44_hm_d_nw_sng` 的哪一输出通道就是 head；UI 框等于原始网络框 |

黄色表示当前选择的对焦区域，白色表示其他检出的候选目标；这些是用户界面和目标选择规则。[DC-S5 官方手册，第 102–103 页](https://eww.pavc.panasonic.co.jp/dscoi/DC-S5/P_PP/DC-S5_DVQP2197ZD_full_eng.pdf#page=102)；[Panasonic 原代 S5 发布稿，2020-09-02，第 3 节](https://na.panasonic.com/news/new-hybrid-full-frame-mirrorless-camera-the-lumix-s5)

**仍未知**：原机识别训练的脸／头标注范围、可见框还是补全框、恢复网络各分类头的真实语义、其他网络是否负责眼或头、原始框到显示框的缩放与后处理。官方功能描述不是 DSP tensor 合约，不能完成这一步证明。

## 4. 四个 NMS 个例的独立核对

四个 train32 个例的人脸 GT 都标记 Occluded=1。同图有几何上包住该脸的 Human head GT，所列 head 均 Occluded=0；这说明原数据确实区分了两种框，但不能用四例代表全部误报。

| 个例 | 被压制候选与该 head 的 IoU | 抑制者与该 head 的 IoU |
|---|---:|---:|
| 1 | 0.57724 | 0.49332 |
| 2 | 0.69903 | 0.40929 |
| 3 | 0.62221 | 0.47803 |
| 4 | 0.43279 | 0.70358 |

前三例的抑制者即使按原始 head 框也未到 IoU 0.5；第四例更符合 head 框。因此，不能把这四个大框统一改判为“正确头部框”。全部同图 head 的原行和几何值已保存，原来的 face TP/FP 不变。

## 5. 相同 64 帧的 face／head 双定义对照

从既有 validation80 排除 12 幅没有 head 框的图片和 4 幅含 head Depiction 的图片，得到同一组 64 帧，含 92 个 face GT、93 个 head GT。对每组原样保存的预测分别一对一匹配 face/head；score=0.5、IoU=0.5 不变。这是一个新的诊断子集，**不能替换或直接混同比较原来的完整 80 帧指标**。

| 保存的模型状态 | Face AP50 | Face TP / FP | Head AP50 | Head TP / FP |
|---|---:|---:|---:|---:|
| 原 K4 权重、原 anchors | 0.37176 | 41 / 38 | 0.55965 | 53 / 26 |
| 解析迁移后的第 0 轮、小 anchors | 0.38930 | 42 / 34 | 0.51324 | 48 / 28 |
| 第 6 轮、小 anchors | 0.42974 | 50 / 185 | 0.42870 | 53 / 182 |

原始恢复模型在该子集更符合 head 标注；这加强了“当前 face 任务与原厂语义可能不完全一致”的假设。以 face 为目标的训练后 head AP 下降，是需要调查的任务偏移，**不是已确认原网络就是 head detector 的证明**。同时，第 6 轮即便按 head 标注仍有 182 个高分 FP，不能靠切换标签定义解释掉训练后的误报激增。

独立检查重新从原 CSV 构建 head 行，核对共同子集、排除规则、所有保存预测的匹配结果、AP 及按 GT 大小分层召回；另用原 `evaluate_detection` 交叉核对 AP。全部通过，无新 forward。报告位于 `artifacts/recognition-ranking-repair/paired-face-head-metrics/independent-verification.json`。

## 6. 后续验证约束

可在具有可靠 head 标注的同一图片子集上，并排评估原输出对 face 与 head 的匹配；必须保持相同预测、阈值和匹配规则，显示各自分母，不选择较高 AP 来替换历史验收标准。这样的对照只能增加或削弱语义假设，不能确认原机等价。

恢复训练之前，应明确本轮要优化的数据任务，并解决定位与评分顺序的验证问题；不能因 head 对照分数较高，就自动把模型改名、改标注或恢复优化。确认原机合约仍需要原机同帧输出、SDK/模拟器，或足以关联原始 tensor 与最终目标类别的固件调用链证据。

机器可读结论及来源哈希见 `artifacts/recognition-ranking-repair/box-contract/report.json`。公开论文和手册的本地只读副本位于同目录 `sources/`。
