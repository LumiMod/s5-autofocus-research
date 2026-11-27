# S5 AF 模型研究交接 · 2026-11-27

**目前没有可部署到相机的新模型；训练已暂停，原始权重与失败实验均保留。**

目标为私有项目 `LumiMod/s5-autofocus-research`；旧仓库及“公开”说明已过时。以下路径相对 `af-research/`。身份识别不在范围内；DFD、镜头及合焦闭环未完成。

## 1. 当前结论

- K4修复后，原权重有图像依赖的检测；旧三种排列不再是有效基线。
- 两次小anchors训练都未通过验收。第二次AP提高，但人脸误报明显增加，小脸仍大量漏检。
- `person_face`只是项目映射，原厂脸／头输出语义未知。最新损失仅验证到冻结输出的梯度，没有新权重或检测改善证据。

详见 [RECOGNITION_TRAINING.md](RECOGNITION_TRAINING.md) 和 [S5_FACE_HEAD_CONTRACT.md](S5_FACE_HEAD_CONTRACT.md)，最新在前；旧章节“未训练”仅指当时。

## 2. 原网络与恢复边界

原组件为 `43_hm_d_nw.bin`、`44_hm_d_nw_sng.bin`。44模型97节点、输入 `[1,1,256,384]`、16×24网格、stride16、25anchors/格；分类各50通道，共享bbox100。仅86/90/94三个1×1头共89,800参数训练，主干逐位不变。

布局 `[O/4,(I×H×W)/4,O4,K4]` 展平I/H/W并跨核边界。跨模型零通道支持此解释；126个卷积索引、往返及合成输出验证通过，bias/upsample未变。

电脑以RGB→Pillow L→双线性拉伸384×256→减117运行浮点图。原机预处理、定点算术、类别映射和DSP数值未验证；菜单QEMU不能证明AF闭环运行。

## 3. 数据、历史命名与固定结果

冻结数据：`private-data/recognition-v2/manifest.json`，SHA256：

`af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8`

train320有883GT（463人/420脸），validation80，test100未推理。小脸为384×256拉伸后面积平方根<32，train/val共103/34个。ID的`validation`指源split；缺纯背景及相机域验收。

同一80张validation；AP50，TP/FP采用score≥0.5、IoU≥0.5的一对一匹配：

| 状态 | 人体AP | 人脸AP | 人体TP/FP | 人脸TP/FP | 小脸/34 |
|---|---:|---:|---:|---:|---:|
| 原K4、原anchors | 60.59% | 34.33% | 70/16 | 47/48 | 0 |
| small-v1：直接换anchors，第6轮 | 42.47% | 10.38% | 71/326 | 30/504 | 1 |
| small-v2：解析迁移，第0轮 | 59.22% | 34.15% | 70/14 | 47/44 | 0 |
| small-v2：第6轮浮点 | 67.59% | 38.30% | 81/64 | 57/215 | 3 |
| 同一第6轮头参数量化代理 | 67.59% | 38.43% | 81/64 | 58/214 | 4 |

两次各6轮/480次AdamW更新，第二次从原K4重启。新scales `[1,2,4,8,16]` 保留ratio/25槽/图；按固定log距离映射原槽 `[3,3,3,9,16]` 并仿射调整bbox，只保持映射槽初始框，不保持全proposal集合。bbox权重初始化19个、最终17个超int16范围；量化代理不模拟DSP。

未过小脸≥17/34、人脸AP≥50%等门槛。固定第6轮为主结果；`recognition-v1/v2`是K4前历史。原权重letterbox的63.20%/38.91%不是拉伸基线。

## 4. 误报、小脸与尚未训练的新损失

train32的10小脸回归覆盖从1升至9，最终仍0：1几何不足、5置信度不足、4被NMS压掉（不是pretop3000或final30截断）。抑制框对脸IoU约0.323/0.294/0.307/0.301，却分数更高。

192次日志RNG重放：73个最终label0人脸FP中67个六轮未抽作负例；这是事后分数追查。小脸/其他脸正例中位数1/16.5；未丢GT但权重不均。头互换证实分类和bbox都影响FP。

`recognition_balanced_loss.py`均衡每GT，最多64难负例加随机补足；回归与同类GT IoU≥0.3的负标签两池都忽略。保护12个正确小脸框，49个可处理FP选中38个，但也漏掉3个主要抑制框。

`recognition_quality_ranking.py`：good为已采正例并有bbox监督；bad无GT归属、质量较低且分数逆序/并列。排除跨GT歧义及无效anchor，几何/选择detach，沿用GT的CE份额。

**最新16项测试、train32的50对冻结输出通过，独立梯度误差≤约7.02e−9，0次更新。** bbox直接梯度不变，但未来全局裁剪可能间接影响更新。两种新loss未接入训练，未证明检测改善。

## 5. 脸／头目标定义阻止继续训练

Human face与Human head是不同源类别，旧manifest只纳入前者，没有扩大脸框。原S5官方资料说明头、脸、眼、身体识别，却未公开当前恢复输出头的对应关系。

64张head子集：原K4 face/head AP37.18%/55.96%，第6轮42.97%/42.87%，head仍182FP。不能替代80帧成绩或证明原厂类别。

28对脸中24对有head参考，7对bad更贴head，3对bad对同一head≥0.5而good<0.5。第4例good脸/头IoU0.563/0.464，bad为0.301/0.704：两任务方向冲突。源CSV无实例关联，最大IoU不证明同一人。

保持 `ready_to_train=false`，等待原厂合约或明确批准的新目标；梯度自检不能解除此门槛。

## 6. 原ARM／QEMU验证到了哪里

原转换链：16字节header+60条24字节记录；低16位xywh→u16 xyxy，分数低16位/65535，保留数字类别0–6而不命名。

QEMU9.2.4 virt/Cortex-A15执行 `0x063bd2c3` 原Thumb指令，60条合成记录的960字节输出+16字节保护逐字节通过；改错1字节的负对照失败。首试DTB冲突保留。这不是整机、CEVA网络或类别语义验证。

## 7. 环境、入口与材料

已验证macOS arm64/Python≥3.12。原生Windows缺`resource`，Linux/WSL需另核；Paddle外部对照依赖另见`requirements-external-baselines.txt`。

可选复核命令，**不是恢复训练**；在新仓库的 `af-research/` 下执行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
S5_COMPONENTS=/path/to/unpacked-v29
.venv/bin/python verify_gemm_k4.py --source "$S5_COMPONENTS" --output artifacts/handoff-check/k4.json
.venv/bin/python s5_small_anchor_target_audit.py --source "$S5_COMPONENTS" --output artifacts/handoff-check/targets.json
```

上述`--source`须为组件目录，不要写`--firmware file`。用新输出路径。单图 `infer_s5_k4.py` 等参数不同，先读`--help`。

| 路径 | 用途 |
|---|---|
| `packing_investigation.py`、`candidate_network.py` | 显式选`gemm_k4`；旧默认值仅为历史兼容 |
| `anchor_head_transfer.py` | 解析迁移API及量化范围；不是原权重原顺序 |
| `artifacts/recognition-small-anchor-v1/`、`recognition-small-anchor-v2/` | 两次训练全部证据；v2固定浮点权重在`epoch-06/checkpoint.pt` |
| `artifacts/recognition-fp-diagnosis/` | FP、采样、抑制者、均衡loss和旧排序结果 |
| `artifacts/recognition-ranking-repair/quality-ranking-normalized/` | 当前排序验证、独立审计、冻结源码 |
| `artifacts/recognition-ranking-repair/`其他子目录 | 脸头合约、双定义对照、ARM/QEMU证据 |
| `evidence/2026-11-27/` | 汇总与哈希索引，不能替代原始产物 |
| `private-data/recognition-v2/` | manifest及源标签；原始照片需另供，保留split和许可 |

完整固件需另供；原组件目录为`s5-v29-offline/unpacked-v29/`。22份checkpoint均为电脑实验产物，不是相机升级包。NPZ、anchors、plan、日志、snapshot、失败记录及合成BIN/harness `.bin/.o/.S`须成组保留；环境可重建。

QEMU主入口`verify_s5_arm_result_converter.py`参数为`--firmware-root/--qemu/--output`，需03_program.bin、verification.json、/usr/bin/clang及parser合成输入/预期文件。历史helper/负对照绑定原路径、QEMU哈希或原分析MD，不是异机通用入口，勿改写历史证据。

**不要直接用旧 `train_recognition.py` 开新训练：它仍仅选择三种历史错误布局。** 两个small训练入口仅供历史复现，未接最新loss。机主协作入口见 [S5机主采集指南.md](S5机主采集指南.md)。

## 8. 后续顺序

1. 核对清单/SHA/依赖，复现证据并保留旧基线。
2. 取得原机同帧输出或SDK，确认proposal/类别/头脸合约；改变任务须明确批准。
3. 核对排序、共享bbox、梯度/裁剪、量化，补背景与相机域验收。
4. 门槛明确后另立有限训练；保留原结构/权重/对照，test100留作最终评估。
