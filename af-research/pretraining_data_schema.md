# 电脑端训练前准备与数据契约

这里说明的准备工具不启动训练、不更新模型。`pretraining_data.py` 校验相机采集、轨迹与物理焦点数据并计算基线指标；`pretraining_readiness.py` 分开报告电脑端候选识别实验条件与原机一致性／部署条件，还能实际重算候选网络与原执行张量对照。通过人工样例测试只代表这些工具可用，不能证明 S5 原模型已还原。识别训练入口及实际实验结果另见项目说明。

## 运行

这是研究者整理后的数据集与原执行张量契约，不要求普通机主直接填写。机主的素材收集入口见 [项目说明](README.md)；JPEG/PNG 标注帧、轨迹和原执行张量由研究者在收件后分别整理。RW2、MOV 和 MP4 原件不能直接充当这里的已标注 `dataset.json`。

在仓库的 `af-research/` 执行；先按 [项目说明](README.md) 安装 Python 3.12+ 的研究环境。以下 macOS/Linux 命令中的 `.venv/bin/python` 在 Windows 对应 `.venv\Scripts\python.exe`：

```sh
.venv/bin/python verify_pretraining_tools.py
.venv/bin/python pretraining_readiness.py --tools-only
.venv/bin/python pretraining_data.py /path/to/dataset.json --output artifacts/data-check.json
.venv/bin/python pretraining_data.py /path/to/dataset.json --predictions /path/to/predictions.json --output artifacts/baseline.json
.venv/bin/python pretraining_readiness.py --firmware-root /path/to/original-s5-v29 --dataset /path/to/dataset.json --predictions /path/to/predictions.json --captures /path/to/original-captures.json
```

没有固件、真实数据与原执行记录时，第二条命令仍可运行，并把缺项写到 `artifacts/pretraining-readiness.json`。`--tools-only` 明确跳过读取固件，模型身份门槛保持 false；它不能与 `--captures` 合用。旧字段 `ready_for_formal_training` 保留原先“原机识别／追踪及物理对焦全部门槛”的含义，不再用它阻断独立电脑识别实验；新调用方应读取明确的 `scopes`。

原执行张量比较需要两份完整、固定 SHA-256 的原始模型组件。`--firmware-root` 接受组件目录或包含 `unpacked-v29/` 的父目录，也可改用 `S5_AF_FIRMWARE_ROOT`；`--source /path/to/components` 优先级最高。历史 ZIP 和完整更新包不是这一步的必需输入。未通过原始模型身份检查时，张量比较会拒绝运行。没有布尔参数可以跳过原执行证据。

## 电脑端人物／人脸检测实验

这个任务使用真实静态图像的人物框与人脸框，不是人脸身份识别。它可以先于完整物理对焦闭环进行；不要求相机、CEVA/CDNN SDK、原 DSP 张量、低光标签或镜头位置。通过条件只是具备开始有界候选模型实验的输入，不能证明网络假设正确、识别性能达标或可上机。

```sh
.venv/bin/python verify_recognition_readiness.py
.venv/bin/python pretraining_readiness.py --firmware-root /path/to/original-s5-v29 --recognition-dataset private-data/recognition-v1/manifest.json --recognition-model 44_hm_d_nw_sng
```

`ready_for_computer_recognition_experiment` 对应 `scopes.computer_recognition_experiment.ready`：检查选定完整原始组件的固定摘要，以及真实图像清单的文件、框、来源组和 train/validation/test 覆盖。`ready_for_original_camera_fidelity_and_deployment` 对应独立 scope；缺失的原执行、预处理、解码、物理对焦及有效加载流程均继续为 false。训练脚本的运行记录另行证明实际训练发生；本只读命令中的 `training_started: false` 仅表示它自身不训练。

公开静态图像使用独立契约，避免为了通过原相机数据契约而编造光线、运动、轨迹或焦点信息：

```json
{
  "schema_version": 1,
  "task": "recognition_detection",
  "classes": ["person", "person_face"],
  "provenance": {"path": "provenance.json", "sha256": "来源记录的实际SHA256"},
  "frames": [{
    "id": "image-id", "path": "images/image-id.jpg", "sha256": "图像实际SHA256",
    "width": 1024, "height": 768, "split": "train",
    "source_kind": "public_dataset", "source_dataset": "数据集名称及版本",
    "source_image_id": "原始图像ID", "group_id": "作者或拍摄序列及重复内容归组ID",
    "boxes": [
      {"class": "person", "bbox_xyxy": [100, 50, 500, 700]},
      {"class": "person_face", "bbox_xyxy": [200, 70, 300, 180]}
    ]
  }]
}
```

每个来源图像 ID 必须唯一，同一来源组和相同文件不能跨划分。验证器对图像完整解码，检查尺寸、SHA-256、原图坐标框、解码 RGB 像素重复及 64 位 dHash 汉明距离 ≤4 的潜在近重复；后者使用灰度双线性缩放至 9×8 的横向差分。数据准备应先按作者／拍摄序列归组，再排除跨划分近重复。此规则可能把相似图片保守剔除，也无法发现所有重复场景；不能把自动检测通过写成无泄漏的绝对证明。

三个划分都需包含所声明类别的标注框。缺少纯背景空框图像会报告 warning，正例子集不能支持全面误报率结论。数据来源、许可、官方原划分、筛选过程和局限保存在摘要绑定的 `provenance.json`；校验器证明文件与结构一致，不独立证明来源真实性或标注完整。未知光线保持未知，不能据此得出低光提升结论。

Python 接口为 `validate_recognition_dataset(path) -> (report, data)`。`verify_files=False` 只可用于诊断，会明确输出 `file_content_verified: false`；准备状态始终调用完整验证，不接受该捷径。训练入口仍需记录候选排列、预处理和解码假设、训练前基线、有限梯度与实际参数更新，并只在 validation 做选择，保留 test 做独立报告。

## 相机采集、追踪与物理焦点数据集

路径相对清单所在目录，禁止绝对路径与越出根目录的 `..`。每个样本校验 SHA-256 和 PNG/JPEG 头部尺寸；这不等于完整图像解码质量检查。`source_kind` 与 `capture_id` 是操作员提供的来源信息，真实性和标注质量仍需审阅，脚本不会声称已验证。

```json
{
  "schema_version": 1,
  "classes": ["person", "person_face"],
  "frames": [
    {
      "id": "capture01-0001",
      "path": "frames/capture01-0001.jpg",
      "sha256": "实际文件SHA256",
      "width": 1920, "height": 1080,
      "source_kind": "camera_capture", "capture_id": "原始拍摄记录编号",
      "sequence_id": "capture01", "frame_index": 1, "timestamp_s": 0.04,
      "split": "train",
      "conditions": {"light": "low", "motion": "moving"},
      "boxes": [{"class": "person", "bbox_xyxy": [100, 50, 500, 1000], "track_id": "person01"}]
    }
  ]
}
```

允许的类别为提取到的六个前景标签：`person/person_face/animal/animal_face/bird/bird_face`。类别之间可有父子框；每个标注轨迹 ID 在单帧内应唯一。框使用原图像素坐标，`x2/y2` 是右/下边界，面积为 `(x2-x1)*(y2-y1)`。负样本必须保留 `boxes: []`。

split 为 `train/validation/test`；同一 sequence 和完全相同文件内容不能跨划分。按拍摄序列划分后还应人工检查近重复帧、同场景重拍及来源重叠，哈希检测不能发现这些泄漏。时间戳需按帧索引严格递增。静态/低光和追踪目标下，训练与验证集都应包含目标类别、低光样本及至少两个连续标注帧的轨迹；这些仅是最低结构检查，不代表数据量足够。

`conditions.light` 为 `low/normal/backlit`，`motion` 为 `static/moving`。低光标签应按采集方案定义并保留照度/曝光信息，不能把任意变暗图片当作真实低光验证。

镜头控制研究可额外为 frame 提供 `focus`：

```json
{
  "lens_id": "实际镜头型号及版本",
  "position": 120, "target_position": 123, "position_unit": "明确的编码器或协议单位",
  "focal_length_mm": 50, "aperture": 2.8, "iso": 3200, "illuminance_lux": 3,
  "ground_truth_method": "焦点包围/标靶与选择标准",
  "trace_path": "traces/capture01.json", "trace_sha256": "实际轨迹文件SHA256"
}
```

目前只验证这些字段和轨迹文件身份，没有证明轨迹对应正确硬件动作。静态/低光合焦仍需要焦点位置真值、镜头驱动及反馈的物理基线；主体框的检测 AP 不能代替这项验证。

没有真实编码器或协议位置记录时，应省略整个 `focus` 对象并保留该项数据缺口。普通照片、屏幕录像或手动转动对焦环，不能据此编造 `position`、`target_position` 或镜头控制轨迹。

有这些数据时，工具会按镜头、位置单位、真值方法和光线条件分别统计位置绝对误差均值/中位数/最大值；不同镜头编码器或不同单位不会合并。该指标不等于图像清晰度、合焦时间或控制器性能。

## 检测/追踪基线

预测清单需包含所有 validation 帧，即使没有检测，也要提供空数组：

```json
{
  "dataset_manifest_sha256": "dataset.json的实际SHA256",
  "model_sha256": "原模型组件SHA256",
  "frames": [{"id": "capture01-0001", "detections": [
    {"class": "person", "bbox_xyxy": [100, 50, 500, 1000], "score": 0.9, "track_id": "tracker01"}
  ]}]
}
```

默认计算 IoU=0.5 下按置信度全点插值的 AP，**不是 COCO mAP**。同类每个真值框只允许匹配一次，重复预测计 FP；并输出 score≥0.5 的 TP/FP/FN。跟踪指标只统计匹配框中的 ID 切换，漏检仍计检测 FN；它不是 MOTA/HOTA。指标计算不会证明预测文件来自原 S5 执行，这由原执行记录/数值对照单独处理。

## 原执行张量契约

这一节属于后续设备/原 CDNN 运行环境的工程捕获任务，普通相机拍摄和录像无法自动导出这些内部张量。没有开发工具或已验证的捕获方法时保留缺项，不要求机主按这里的内部地址进行操作。

必须知道采样点确实位于每个卷积的激活前。若原执行输出已融合 ReLU，不能把它标作激活前来凑比较；应另外实现该采样点的正确比较器。当前工具明确拒绝未知采样点。

`original-captures.json`：

```json
{
  "schema_version": 1,
  "captures": [{
    "id": "real-run-001", "model": "43_hm_d_nw",
    "model_sha256": "模型组件实际SHA256",
    "capture_kind": "physical_s5",
    "capture_point": "convolution_pre_activation",
    "device_or_runtime_identity": "机身固件/硬件或原CDNN运行库版本",
    "capture_method": "实际导出命令及步骤",
    "captured_at": "ISO8601时间",
    "run_log": {"path": "captures/run001.log", "sha256": "实际SHA256"},
    "reference_npz": {"path": "captures/run001.npz", "sha256": "实际SHA256"}
  }]
}
```

`capture_kind` 只接受 `physical_s5` 或 `original_cdnn_runtime`。NPZ 包含 `prepared_input`，以及所有卷积节点的 `conv_<层编号>`；例如 `conv_1` 是 conv1 激活前输出。输入为恢复实数单位的 float32 NCHW `[1,1,256,384]`，每个输出 shape 必须与图一致。不能省略难以采集的层而让部分对照冒充整网。

脚本实际重算三种候选排列，逐层比较 `atol=2e-5, rtol=2e-4`，不是读取“passed=true”。每份模型至少两组不同输入，且仅一个排列通过全部采样，才通过数值比较门槛。这是浮点移植误差检查，不是定点逐位一致证明。文件摘要绑定运行日志与张量，但日志的外部来源仍需审阅；任何本机脚本都不能仅凭 JSON 证明它确实来自相机。

当前剩余三个门槛没有自动通过入口：真实图像到模型输入的预处理对照、proposal 解码/输出 ABI 对照、物理合焦控制基线。以后须加入具体可复现的证据验证器，不能添加手填 `true` 代替验证。

[templates/](templates/) 提供三个相机采集／原执行空清单，可复制后填充。空数据或占位摘要会被拒绝。`data_and_evaluation_tools_prepared` 仅在工具的样例测试通过、测试报告绑定的源文件摘要仍一致时为 true；它与两个任务 scope 分开，不代替真实识别数据或原机验证。
