# 电脑识别试验的真实数据

`recognition_data.py` 下载少量有人工框标注的 Open Images 图片，用于 S5 候选网络的人物／人脸**检测**试验。人脸类别表示框出人脸，不是识别某人的身份。数据与固件、模型 checkpoint 分别留在本地忽略目录，不上传人像。

## 来源和使用条件

- [Open Images V7 官方下载页](https://storage.googleapis.com/openimages/web/download_v7.html) 链接的 V5 validation/test 框标注（V7 延用这些框）、V7 类别表，以及官方 image metadata。
- 图像来自官方脚本使用的 [CVDF 公共图片桶](https://github.com/openimages/dataset/blob/master/downloader.py)，按 image ID 单张下载；不下载整个数据集。
- [官方说明及许可](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)：标注为 CC BY 4.0；图像元数据列为 CC BY 2.0。脚本仅接受具有该图像许可记录的样本，并逐图保留作者、标题、来源页面和许可链接。没有重新逐页审核 Flickr 当前许可，不能把官方元数据当成独立法律核验。
- 学术引用：Kuznetsova et al., *The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale*, IJCV 2020。

默认下载 320 张 train、80 张 validation、100 张 test。这里的 train/validation 均来自 **Open Images 官方 validation**，test 来自 **官方 test**；这是单独命名的小规模实验划分，不是官方基准成绩。

## 筛选和划分

1. 从官方层级表找出 Person 及其 Subcategory（Man、Woman、Boy、Girl），统一映射为 `person`，Human face 映射为 `person_face`。不使用年龄或性别分类目标；保留原始标签 ID 仅供审计。
2. 要求每图两类均有框。官方 validation/test 对已有正标签的实例提供完整框标注；要求两类有框，可减少把未标注的类别当背景的风险，但不能消除标注遗漏。
   抽样框可视化发现，Girl/Woman 或 Boy/Man 有时给同一人物两个边界略不同的框；后续检查也发现 IoU≈0.64 的 Girl/Person 重复。统一为 `person` 后会重复计数。因此不同源人物子类的框 IoU≥0.5 或交集／较小框面积≥0.95 时，排除整图，不猜测哪一个框更正确。该保守规则也会去掉部分真实的嵌套／重叠人物，结果不代表这类场景的表现。修正由标注 QA 触发，不使用模型预测或得分筛选。
3. 目标框中只要有 group-of 或 depiction，就排除整张图片。保留遮挡、截断和 inside 标记。不按框大小、图像亮度或模型得分筛图；没有挑出“模型容易识别”的图片。
4. 缺失作者、非 CC BY 2.0 元数据、非零旋转元信息或非平凡 EXIF 朝向的图排除，避免来源隔离与框旋转产生歧义。
5. 规范化 Flickr 作者 profile URL 形成 `group_id`；固定 seed 和 SHA256 将作者分配到 train、validation、test。每位作者最多两图。同作者的相邻拍摄或连续上传不会跨划分。图像按固定 SHA256 排序，达到每个划分的数量后停止。
6. 逐图记录文件 SHA256、解码 RGB 摘要，以及 64 位 dHash。相同文件、相同像素、dHash 汉明距离 ≤4 的后续图从所有划分中剔除。dHash 使用灰度 9×8、Pillow BILINEAR resize、相邻右像素大于左像素。

没有可靠的拍摄序列或拍摄时间元数据，因此作者分组是保守的替代方式，不称为已核验的拍摄序列。跨作者转载、裁切或复杂变换的近重复仍可能漏过；自动审计不能证明所有图内容独立。剔除原因保存在 `provenance.json`。

测试独立性针对本次微调。原厂模型最初使用的私有训练集未知，无法排除它与公开图片有交集。

**范围限制：**这是正例图像的小样本检测试验，未测无人物场景的整图误报。背景区域仍用于检测训练，但不能替代整图负样本验证。没有真实光照条件、S5 域内图像、镜头状态或对焦轨迹，因此结果不能支持弱光改进、实际合焦能力或相机部署结论。

## 复现

在 `af-research/` 内使用已安装 Pillow 的 Python；不需要下载新的训练框架：

```sh
python recognition_data.py --output private-data/recognition-v1 \
  --train 320 --validation 80 --test 100 --seed 20261127 --workers 5
```

输出包括 `raw/` 官方元数据、`images/` 原始下载文件、`manifest.json`、`provenance.json`、`data-summary.json`。原始 CSV 与每张图片均记录 SHA256；manifest 绑定 provenance 的 SHA256。下载失败和数据筛选排除记录保留，首次取得的 manifest 是本次实验的冻结输入。

如需在另一目录重建**完全相同**的样本集合，携带冻结的 manifest 与它引用的 provenance，执行：

```sh
python recognition_data.py --from-manifest /path/to/frozen/manifest.json \
  --output private-data/recognition-replay
```

此模式不重新抽样，按冻结 ID 下载并逐图校验原 SHA256；上游内容改变或消失会失败，不能用新的图片替换却继续称为同一试验。

manifest 使用 `task: recognition_detection`、`classes: [person, person_face]`；`frames` 中保留 `id/path/sha256/width/height/split/source_kind/source_dataset/source_image_id/group_id/boxes`。`bbox_xyxy` 是原图像素边界，右下边界不额外加一。真实光照和轨迹未知时不填相关字段。

## v2：旧 test 已看过结果后的新留出集

`prepare_recognition_v2_data.py --previous private-data/recognition-v1 --output private-data/recognition-v2` 保留原 320 train、80 validation 的完整记录和图像字节，另从 official test 准备 100 张未用于推理的新图。所有旧 500 张的 ID、作者、文件 SHA、解码像素 SHA 及 dHash 距离≤4 的图均排除，新图之间继续去重，每作者最多两图。旧文件使用硬链接节约存储，原图应保持只读用途。

当前 v2 manifest SHA256 为 `af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8`；新 test 有 157 个人体框、147 个人脸框、98 个作者组，与旧 500 图的最小 dHash 距离为 10，新增图片约 30.2 MB。它仍全部为含人体和人脸的正图；没有增加 S5、弱光或背景负样本覆盖。完整旧清单保存在 `ancestry/`，仅用于来源审计。

v2 六轮目标修复实验未通过 validation 有效性筛查，因此这 100 图尚未被模型推理或计算 AP。文件／标签 QA 不等于模型验收；旧 v1 test 已用于诊断，后续不再称作独立测试。
