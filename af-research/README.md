# S5 对焦研究与数据采集项目

面向第一代 **Panasonic LUMIX DC-S5**，同时准备主体识别/追踪和静物/弱光对焦研究。项目包含模型解析、候选网络实验、数据评估接口、机主采集指南和文件交接工具。

**电脑端候选识别训练与原机适配分别推进。** 两份网络的整数参数已提取，K4权重布局已恢复明显的图像识别信号。在小尺度候选框和共享回归目标通过检查后，已完成两次有限训练；整体检测改善，但小脸效果仍未达到验收要求。原厂DSP数值一致性及机内适配仍未验证；电脑checkpoint不是可刷回相机的固件。

## 电脑端人物／人脸训练验证

训练入口、数据来源、复现命令和结果见 [识别训练记录](docs/RECOGNITION_TRAINING.md)，数据筛选与许可见 [数据说明](docs/RECOGNITION_DATA.md)。原始固件、真实照片、预测预览和 checkpoint 均留在忽略目录，不提交仓库。

**最新定位排序核查：** 新增不改变网络的排序监督，通过16项测试和train32的50对冻结输出梯度检查，可覆盖此前漏掉的3个高分定位不足框。但在24对有head参考的人脸候选中，3对存在脸／头目标方向冲突；原`person_face`的工厂语义仍未确认，因此损失未接入训练、权重未更新。另用QEMU实际回放并核对了原ARM结果转换函数，尚不等于执行DSP网络。见[本轮结果](docs/RECOGNITION_TRAINING.md#定位排序与脸头定义核对2026-11-27未恢复训练)和[脸／头定义核查](docs/S5_FACE_HEAD_CONTRACT.md)。

**最新误报诊断（未继续训练）：** 原K4人脸TP/FP为47/48，迁移初始化为47/44，第6轮为57/215。保存预测和实际抽样重放表明，高分错误框大量未受负例监督，小脸正例影响也偏低；四个已有合格高分框的小脸被更高分、定位更差的框抑制。已实现独立的按GT均衡、带正确回归框保护的难负例损失，并通过冻结输出梯度检查，但它尚未覆盖全部定位和小脸问题，因此`ready_to_train=false`，网络和现有权重不变。详见[本轮排查](docs/RECOGNITION_TRAINING.md#误报与小脸联动诊断2026-11-27继续暂停训练)。

**最新小尺度训练：** 保持原97节点图、16×24输出网格、每格25个anchors和50/50/100通道，尺度改为`[1,2,4,8,16]`。13项目标语义测试及全部883个训练GT检查通过；先前直接换尺度会破坏head通道含义，已用固定通道映射和bbox仿射迁移修正初始化。两次各6轮/480步，只训练原三个头的89,800个参数，骨干未变。第二次固定第6轮：人体/人脸AP50 **67.59%/38.30%**，小脸 **3/34**；头部量化代理为67.59%/38.43%、小脸4/34，仍未达到17/34及人脸AP50≥50%的门槛。没有继续加轮、放宽阈值或评估预留test。见[本轮设计与结果](docs/RECOGNITION_TRAINING.md#小尺度候选框与共享回归训练2026-11-27)及[机器可读摘要](evidence/2026-11-27/small-anchor-training/summary.json)。

2026-11-27：已修复分类与共享框回归的 GT 冲突，22 项工具测试及全部 train/validation 目标审计通过。v2 已实际重跑 6 轮／480 次更新，但 validation 平均 AP50 从原始 17.05% 降到最后一轮 13.02%，best 仍是 epoch 0；原始候选的真实图成绩还低于灰图对照。当前不扩大训练，新 100 张 test 保留未推理。汇总见 [v2 验证结果](evidence/2026-11-27/recognition-v2/summary.json)。

第一轮预处理、权重排列、候选框与定点幅值排查未找到有效修复，历史证据保留在 [初轮诊断汇总](evidence/2026-11-27/recognition-diagnosis/summary.json)。随后 v3 从两份原模型的零通道关系中定位到新的 **`gemm_k4` 排列：展平 I×H×W 后每4个元素打包**，它不在旧720种六轴排列中。同一80张 validation 上，原权重人体／人脸 AP50 升至 **60.59%／34.33%**，灰图为0.85%／0.22%；目标平移与遮挡检查也支持图像依赖。保持比例的电脑缩放另测得63.20%／38.91%，没有据此宣称恢复了相机预处理。

K4原权重阶段的小脸基线为0/34，详见 [K4修复记录](docs/RECOGNITION_TRAINING.md#k4权重布局修正2026-11-27未训练) 和 [历史v3证据摘要](evidence/2026-11-27/recognition-diagnosis-v3/summary.json)。`infer_s5_k4.py` 仍是原权重单图推理入口。旧 `train_recognition.py` CLI只比较三种历史布局，**不得直接用于K4训练**；本轮使用独立的`train_small_anchor.py`和`train_small_anchor_transfer.py`，各有冻结的预检、配置和源码快照。

外部参考已实跑：同一80张验证图上，PicoDet-S人体AP50为88.41%；BlazeFace原图／384×256的人脸AP50为63.51%／73.59%，34个参考小脸分别命中25／19个。三组真实图均明显超过灰图与错图，保存预测的独立复算通过。按用户要求，这两个模型只作电脑对照，后续仍走原S5网络路线；**外部通过不解锁原网络训练，也不证明新模型能在机内运行**。见 [结果与复现命令](docs/RECOGNITION_TRAINING.md#picodetblazeface-外部基线已验证2026-11-27未训练) 和 [小报告](evidence/2026-11-27/external-baselines/summary.json)。

`pretraining_readiness.py --recognition-dataset private-data/recognition-v2/manifest.json --firmware-root /path/to/original-components` 分别报告电脑实验与原机适配条件。电脑实验不以 DSP 对照或镜头控制数据为前提；`ready_for_formal_training` 仅保留为完整原机目标的旧兼容字段。输入条件通过不表示上面的识别有效性检查通过。

## 发给 S5 机主

直接转发 [这段消息](docs/给S5机主的转发消息.md)，附上 [完整采集指南](docs/S5机主采集指南.md) 或 [下载机主采集包 ZIP](dist/S5-owner-collection-kit.zip)。ZIP 内含说明、空目录、记录表和可选校验工具；机主无需安装开发环境。

第一轮只需约 15 分钟试采：

| 要交的内容 | 用途 |
|---|---|
| 机身/镜头版本及设置照片 | 保留测试条件 |
| 正常光、暗光各 6 次静物 RAW+JPEG | 检查重新合焦与失败情况 |
| 走近退远、暗光走动、横移遮挡各 1 段原视频 | 观察跟焦、遮挡恢复与拉风箱 |
| 同时用手机拍相机 LCD | 记录可见对焦框、提示和操作时序 |
| 正常光、暗光各约 7 张手动对焦参考 | 比较同构图下可达到的清晰度 |

使用相机正常功能即可。保留失败样本；不知道的参数留空。普通机主不需要采集内部 DSP 张量或镜头编码器。收到试采并核对后再扩大数量。

**这是独立的私有研究交接仓库。** 原始数据照片和完整固件由接手者按来源说明另行准备；采集工具不自动上传文件。

## 整理采集文件

Python 3.10+，只用标准库；不需要固件、QEMU、NumPy 或 PyTorch。在本目录执行：

```sh
python3 collection.py init sessions/pilot01 --session-id pilot01
# 按编号复制相机/手机原文件，填写实际设置后：
python3 collection.py inventory sessions/pilot01
python3 collection.py check sessions/pilot01
python3 collection.py pack sessions/pilot01 --output private-data/pilot01.zip
```

详细步骤见 [采集工具说明](docs/COLLECTION_TOOLS.md)。检查范围是目录、文件摘要和事件引用；不会解码全部媒体、自动标注或证明真实合焦。研究者再按 [数据契约](pretraining_data_schema.md) 整理标注、预测和原执行对照。

## 研究代码：安装和复现

研究环境需要 **Python 3.12+**。以下命令在 `af-research/` 内运行；Windows 激活虚拟环境后把 `.venv/bin/python` 换成 `python`。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_collection.py
# 无需固件：运行人工样例测试、候选框自检及保守的训练就绪检查
.venv/bin/python prepare_offline.py --tools-only
```

完整模型实验需要在本地准备原始 V2.9 中的两份完整组件，代码会验证固定摘要。原固件和模型权重不包含在这个子项目的上传内容中；不会从网络自动下载，也不把仓库中的实验补丁固件当作原件。

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `43_hm_d_nw.bin` | 4,194,304 字节 | `d74e1f38fa7ce7a40ed49e6ffc3fd65bae70ae77eb9d37add2f896abc52897d6` |
| `44_hm_d_nw_sng.bin` | 4,194,304 字节 | `b510989f1e12fd1e6a2efbc5306b125289cb796c48e5ffefe7e4d4ce26390a50` |

```sh
# 输入目录直接包含上述 .bin，或其下包含 unpacked-v29/
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-components
# 同时用一张本地图片检查两模型 × 三种排列的图片流程
.venv/bin/python prepare_offline.py --firmware-root /path/to/original-components --photo /path/to/photo.jpg
# 或统一配置所有研究脚本的输入位置
export S5_AF_FIRMWARE_ROOT=/path/to/original-components
.venv/bin/python photo_probe.py /path/to/photo.jpg --model 44_hm_d_nw_sng --compare-layouts
```

也支持 `--source` 直接指定组件目录。ZIP/TSV、原更新包和解密清单是可选交叉证据；缺失明确记录为未核对，存在却不匹配会拒绝。若同时提供 `03_program.bin`、`39_hm_c_prog.bin`、`40_hm_d_prog.bin`、`41_hm_c_ddr.bin`、`42_hm_d_ddr.bin`，还会运行加载链/运行库调查；否则明确记录跳过该步骤。

运行产物保存在 `artifacts/`，默认不提交。`computer-preparation.json` 保存各步骤结果、日志摘要和源文件摘要。整个 `prepare_offline.py` 流程只做解析、前向、梯度检查和校验，**不执行优化器更新**。旧 `probe_first_conv.py` 单独运行会做一次人工损失演示，故不在准备流程内。

## 证据、边界与下一步

- [本次复现记录](evidence/README.md)：成功运行的范围、摘要与缺项。
- [研究笔记](docs/RESEARCH_NOTES.md)：参数恢复、候选图、三种权重排列与现有证据。
- [算子调查](ops_investigation.md)、[运行库调查](runtime_investigation.md)：已确认字段和仍待原执行验证的解释。
- [数据和评估契约](pretraining_data_schema.md)：标注、AP50、轨迹、物理焦点数据与原运行张量的边界。

机主试采将用于后续相机域内验证、追踪和真实合焦基线。公开真实数据上的识别实验可以先推进；原网络数值还原仍需要可靠的原执行输入／输出或对应开发工具，普通照片不能补齐。原机条件缺失时，`ready_for_original_camera_fidelity_and_deployment` 与旧兼容字段 `ready_for_formal_training` 继续为 false，不否定电脑候选训练已经运行。

## 开发检查与重新生成机主包

```sh
python3 verify_collection.py
python3 build_owner_kit.py
```

采集工具测试只用人工文件验证完整性流程，不将人工样例算作相机性能数据。机主 ZIP 可由已提交说明和脚本重新生成，内含空白模板，不含采集素材。

仓库内另有 [GitHub Actions 配置示例](ci/af-collection.yml.example)。当前上传令牌不含 `workflow` 权限，故未启用远端 CI；有相应权限的维护者可将它复制为 `.github/workflows/af-collection.yml`。本次复现证据来自实际本地运行。
