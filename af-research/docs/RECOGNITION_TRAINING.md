# S5 候选网络：人物／人脸训练与修复验证

本实验使用 S5 V2.9 的 `44_hm_d_nw_sng` 候选计算图与恢复权重，在真实图片上做人物／人脸检测。人脸表示定位人脸框，不涉及身份识别。未使用教师模型训练，也没有用另一个成熟网络替代 S5 网络。原厂数值等价和相机适配仍未验证。历史 v2 是修复目标分配后的有限实验；当前 v3 继续排查权重、预处理和小脸，外部模型仅作对照。

## 定位排序与脸头定义核对（2026-11-27，未恢复训练）

本轮保持原97节点计算图、50/50/100输出通道、共享bbox和所有现存权重，优化器更新为0。新增的是训练损失候选及原始结果格式验证工具，不是一个检测效果已经改善的新checkpoint。预留test未读取或推理。

**纠正此前的任务假设：`person_face`不能仅凭名字直接等同于Open Images的Human face。** 数据源另有Human head类，通用规范是框住目标的可见范围。原代S5官方资料明确分别识别头、脸、眼、身体，但未公开它们与当前两个网络输出头的映射。完整来源、类别MID、过滤规则和缺口见[脸／头定义核查](S5_FACE_HEAD_CONTRACT.md)。这份说明没有把头框当成人脸真值，也没有改写历史成绩。

在同一64张有可靠head标注的验证图片上，原K4对face/head的AP50分别为37.18%/55.96%，第6轮为42.97%/42.87%。原模型更符合头部标注是一个行为线索，尚非原厂类别定义的证明；训练后即使按head衡量仍有182个高分FP（原26），因此不能用换标签消除误报问题。完整80张历史结果与新的64张诊断结果分别保留。

### 新排序监督已通过哪些检查

`recognition_quality_ranking.py`保留旧GT所有者，不新增网络层。每个GT最多选择一对候选：good必须是已采样、由该GT拥有且有bbox监督、IoU≥0.5的正例；bad必须未被任何GT拥有，静态标签为负／忽略，IoU在[0.3,0.5)，与good相互重叠且分数逆序或并列。与其他同类GT有歧义的候选、无效anchor、已经正确回归的框均不进入bad池。选择与IoU脱离计算图，只有分类logits收到排序梯度。

排序项是`ΔIoU × softplus(z_bad − z_good)`，`z=前景logit−背景logit`，无新增margin。每对的权重使用该GT在基础分类CE中的正例权重份额，未配对GT的份额不转给其他GT。归一化前的探索报告和源码快照完整保留；当前版本按类检查总pair权重不超过基础正例CE份额，质量加权后不超过该份额的0.7。**这约束权重总量，不保证排序梯度一定小于已经很小的CE梯度。**

16项合成边界／梯度测试，以及固定train32的50对真实输出检查通过；独立标量实现的最大损失／梯度误差约7.02e−9。此前3个被guard忽略的主要小脸抑制框，现在得到正确的下压梯度；另一例仍由基础难负例监督处理。共享bbox的直接梯度逐元素不变。所有梯度只在冻结模型产生的输出副本上计算，没有模型参数更新。

不能把上述结果当作小脸召回已经提高。另有全局梯度裁剪耦合：旧runner共同裁剪三个头，未来若加入排序项并触发裁剪，bbox实际更新也可能间接受影响，不能承诺“新增排序完全不影响回归更新”。当前损失未接入任何训练入口。

### 新发现的目标冲突，仍阻止继续训练

50对中有28对人脸候选，24对有可用head参考。其中7对bad更贴近head标注，3对出现“bad对同一head的IoU≥0.5，而good<0.5”。例如第4个小脸案例中，实际选中的good对脸／头IoU为0.563/0.464，bad为0.301/0.704。按可见脸任务，排序方向正确；若原厂输出承担头部候选任务，这个方向反而可能损害它。官方CSV没有face/head人物实例关联，报告保留全部同图几何，未把最大重叠当作已确认身份对应。

因此，当前修复是**可见脸任务下数学正确的损失候选**，不是已确认适用于原S5目标的升级。仍需把原始proposal、数字类别及最终头／脸用途串起来，或获得同帧原机原始输出／SDK对照。现有误报、小脸漏检的实测验收未通过，`ready_to_train=false`。

### 原固件结果边界的实质进展

已追到原ARM结果链及后续复制／坐标处理，选定路径没有按class2将头框专门缩为脸框的规则；这只排除这些已检查段，不能替代DSP输出生产者的语义证据。独立解析器按原指令恢复16字节header加60条24字节记录：低16位xywh转为u16 xyxy，分数低16位除以65535，数字类别0–6原样保留；无类别名称推断。

`verify_s5_arm_result_converter.py`用本地QEMU 9.2.4的virt/Cortex-A15环境执行原地址`0x063bd2c3`的未修改Thumb指令和常量区。60条合成记录的960字节输出及16字节尾部保护逐字节一致；仅破坏预期输出1字节的负对照按预期失败。首试因stub与virt DTB重叠在CPU运行前被拒，原失败输出保留；移动stub后成功。这不是相机整机、CEVA DSP或神经网络执行，不能证明原识别模型数值等价。

新证据位于`artifacts/recognition-ranking-repair/`；[可提交汇总](../evidence/2026-11-27/recognition-ranking-repair/summary.json)绑定合同、双标注对照、排序检查、独立审计及原ARM回放。旧误报诊断、所有训练checkpoint和失败实验均保留。

## 误报与小脸联动诊断（2026-11-27，继续暂停训练）

用户要求查清误报增加、保持原网络结构，解决并验证后再训练。本轮优化器更新为 **0**，不改变旧推理阈值、NMS、anchors、网络图和现存checkpoint；预留test仍未推理。新增修复仅为独立、尚未接入训练入口的损失候选，**监督检查通过不等于误报或小脸检测已经修好**。

### 误报增加发生在哪里

对同一80张validation已有的13,818条检测逐框复算，两套独立实现的匹配、AP、几何和互斥分类完全一致。固定score≥0.5、IoU≥0.5：

| 已保存阶段 | 人体TP / FP | 人脸TP / FP |
|---|---:|---:|
| 原K4、原anchors | 70 / 16 | 47 / 48 |
| 迁移初始化、未训练 | 70 / 14 | 47 / 44 |
| 迁移训练第6轮 | 81 / 64 | 57 / 215 |

最终215个人脸FP中，138个与同类GT有0.1–0.5的IoU（定位不足），68个同类重叠更低，7个与另一类GT重叠≥0.5，2个为重复匹配；类别按重复、跨类、定位、低重叠的优先级互斥划分。低重叠只描述与现有标注的关系，不能直接断言画面中没有真实人脸。AP提高的同时，固定阈值人脸precision由49.5%降至21.0%；增加的误报不只来自小框，也不是主要来自重复框。

冻结train32做分类头／回归头互换：相同新anchors下，初始化人脸TP18/FP19；只换训练后分类头为23/71；只换训练后bbox头为20/50；两者都换为23/83。该反事实说明评分更新与回归几何都有作用，不能只归因于换anchors或全局分数校准。混合头仅是诊断状态，没有保存成升级模型。

### 实际训练抽样与小脸丢失链

对train32的192次真实图片训练日志，逐次从记录的RNG状态重放原采样，并核对下一状态与日志计数。最终73个label0高分人脸FP中，67个在六轮里从未被抽作负例。这个结论用**训练后的分数追查原先抽样**，不声称知道每个历史step的当时分数。小脸每次访问得到的正样本中位数为1，其他脸为16.5；GT虽未漏掉，监督影响仍不均衡。

10个小脸中，1个缺少IoU≥0.5回归框，5个已有合格框但分数不足，4个已有合格高分框却被NMS实际抑制。后4个不是pretop3000或最后30框截断：它们的最高分合格候选被评分更高、IoU仅约0.323/0.294/0.307/0.301的错误框直接压掉。盲目增加最终框数不能解除这些已发生的抑制。

逐图检查这完整4例还发现：保留框大多覆盖较完整头部／帽子，标注框更贴脸部区域，4个目标均有`IsOccluded=1`。这里的“错误”是相对当前标注的IoU定位失败，**不等于没有识别到人，也不能当作纯背景假阳性**。原厂训练的face/head框范围仍未知；证据图位于`artifacts/recognition-fp-diagnosis/nms-evidence/comparison.png`，未修改匹配标准或把这些FP重新计成TP。下一步处理定位排序前，要区分真正背景错认、框范围不一致和确实偏离脸部，不能仅按现有分数压掉所有完整头部框。

另有一个必须保护的反例：原anchor标签为0的候选，回归后可能已经正确框住GT。固定train32有12个这样的正确小脸候选，其中2个score≥0.5。无保护地对高分label0做难负例挖掘，会反向压低这些正确框。

train320标注审计确认883个源目标框均保留，没有发现导入漏框；不能由此宣称源标注绝对完整。静态跨类潜在冲突仅占实际人脸负例抽样的0.283%，也不能单独解释此次大幅增长。数据仍缺少纯背景图片和相机域验证。

### 已实现并验证的有限修补

新增`recognition_balanced_loss.py`，原`train_recognition.py`和两次训练源码保持可复现：

- 保留同一GT所有者及每个GT的强制覆盖，在原32/128预算内均衡正样本；分类和bbox的每GT权重相等，总样本权重不增加。
- 每类最多64个难负例，其余预算随机抽样，直接针对高损失背景候选。
- 静态label0但回归框与本类GT IoU≥0.3时，既不进入难负例，也不进入随机负例；只忽略该分类监督，不偷偷改成正例或另分配bbox所有者。

`verify_balanced_loss.py`的6组边界/梯度fixtures及冻结train32的真实输出检查通过：覆盖、预算、唯一所有者、等GT权重、难负例排序和梯度方向正确；所有梯度只对脱离模型的输出张量计算，模型参数无梯度、无更新。该模块尚未被任何训练入口调用。

独立复核进一步确认：49个符合新负例条件的人脸FP，本次选择38个（37个hard、1个random），另11个未选；12个已正确回归但静态label0的小脸候选全部被保护。83个FP完整划分为38个入选、24个被动态保护、10个静态ignore、11个符合条件但未入选。这是监督覆盖改善，不能当作实际FP由83降为45。

**仍未解决：** 保护规则只能直接针对83个人脸FP中的49个低IoU、静态负例；四个小脸最高分错误抑制者中只有一个进入这个范围，另三个会被保护性忽略。其余定位不足、低置信度和一个几何缺失的小脸，没有经过新权重的效果验证。下一项是验证定位质量与分类排序的联合监督，且不得把回归已正确的候选重新压成背景；不能凭上述监督自检开启长训练或宣称检测通过。

详细报告位于`artifacts/recognition-fp-diagnosis/`的`saved-predictions`、`head-interactions`、`training-sampling`、`target-conflicts`和`balanced-loss-verification`；可提交[汇总](../evidence/2026-11-27/recognition-fp-diagnosis/summary.json)仅包含数值与文件摘要，不含照片或权重。

## 小尺度候选框与共享回归训练（2026-11-27）

用户要求先验证小尺度候选框和共享回归目标，再开始训练。本轮已按此顺序完成两次有限实验，网络结构保持原样：44模型97个图节点、384×256单通道输入、16×24特征网格、stride16、每位置25个anchor、分类头各50通道、共享bbox头100通道。只训练原86/90/94三个1×1头的89,800个参数，主干逐位不变。**训练已实际完成，但小脸效果仍未通过验收，不能称为已解决或可用于相机。**

### 训练前检查

仅用320张train的标注比较三套尺度，保持原5种ratio及槽位数。选定`[1,2,4,8,16]`：IoU≥0.5的小脸anchor覆盖从0/103提高到48/103，大人体保持374/376、大脸保持111/111；中等人体覆盖少4个、中等脸少3个。更激进的`[0.5,1,2,4,8]`只多覆盖1张小脸，却大量损失大目标，因此未采用。共同stride16仍限制中心位置匹配；覆盖率并非回归后检测率。

共享目标审计包含13项语义测试，以及全部883个训练GT、3个固定采样种子共960次合成预测梯度检查。每个正anchor只有一个GT所有者，分类和共享bbox都指向该GT；无跨类回归冲突、零IoU强配、采样丢GT或已采样分类正例缺少bbox监督。低IoU强配明确保留：仍有55/103小脸低于0.5，不能把“存在监督目标”称为“已能检测”。两proposal中的scale数组各20字节，在内存修改、重新解析和完整逆还原通过；97节点图、权重、偏置、ratio及输出形状未改变，没有保存修改后的固件BIN。

第一试验直接沿用原head槽位、替换anchors。新anchor epoch0的人体/人脸AP50跌至0.058%/0.140%，说明虽然模型能执行，槽位的预训练尺寸含义已经改变。6轮后仍退化；此试验完整保留，没有继续加轮或作为原模型升级。

随后修正初始化：按与数据无关的log尺度距离，将新`[1,2,4,8,16]`映射到原`[3,3,3,9,16]`，分别复制每ratio的bg/fg分类通道；bbox的xy权重/偏置乘原尺寸与新尺寸之比，wh偏置加该比值的log。合成9600行核验了映射槽的框等价，FP32最大坐标差约6.9e−5像素。它只保留每ratio的3种独立原槽，原scale6/32不保留，重复槽也会影响候选竞争，**不是完整原proposal集合的等价替换**。固定train32对照中，人体/人脸AP50从直接换anchors的0.117%/0.102%恢复到71.28%/35.87%，明显超过灰图与错图；原anchors对照为74.61%/40.04%。策略没有根据validation搜索或修改。

### 固定训练结果

两次均从原K4恢复权重开始，第二次加上述解析迁移，不续训第一次的checkpoint。每次6epochs、320图/epoch、4图梯度累积、80steps/epoch，共480次AdamW更新；学习率1e−4、weight decay1e−4、梯度范数上限10。固定第6轮为主结果，另记录包含epoch0的最佳验证平均AP对应轮次，仅作辅助。阈值不变：score≥0.5、IoU≥0.5、一对一匹配后统计34个参考小脸。两轮合计960次真实更新、3,840次训练图访问。

| 固定80张validation结果 | 人体AP50 | 人脸AP50 | 小脸命中/34 |
|---|---:|---:|---:|
| 原K4权重、原anchors | 60.59% | 34.33% | 0 |
| 直接换anchors，第6轮 | 42.47% | 10.38% | 1 |
| 迁移初始化、未训练 | 59.22% | 34.15% | 0 |
| 迁移初始化，第6轮FP32 | **67.59%** | **38.30%** | **3** |
| 同一第6轮头部量化代理 | 67.59% | 38.43% | 4 |

正式冻结门槛包括小脸召回≥0.5（至少17/34）、人脸AP≥max(0.5,原基线+0.05)、人体AP下降不超过0.05、平均AP提升≥0.02、两类真实图各比所有对照高≥0.15。第二轮通过人体保留、平均提升和真实图对照检查，但人脸AP及小脸召回均失败；不能用整体涨分代替小脸通过。全部保存分数下，第二轮FP32最终框能匹配10/34小脸，score≥0.5只匹配3个；量化代理分别11个和4个。

固定score0.5下，全部人脸TP由47升至57，同时FP由48增至215，置信度与误报表现明显恶化。新增3张小脸参考尺寸约32、32、22像素；9张小于16像素的脸仍全部漏检。对照图中的有框不能代替IoU匹配或误报检查。

量化代理只对三个头的weight/bias按原scale做float64除法、ties-to-even取整、整数范围clip，再以float64乘scale转float32回放。它不模拟激活、累加器、舍入顺序或DSP。第二轮初始化有19个bbox权重越界，最终仍有17个；分类权重和bias无裁剪。其余数值误差与独立重建一起记录。原始weight整数回归逐字节相同；bias经过float32会丢失少量原i32低位，报告明确区分原字节往返与float回放，未假装原bias整数无损。

第一轮训练结束后的报告步骤曾因错图框y2=768.0000000000001触发严格边界检查而退出。所有480次更新和第6轮checkpoint已保存；独立finalizer仅夹紧≤1e−9像素的误差，保留原失败JSON，逐项核对26个训练产物未变化，额外optimizer更新为0。第二轮在冻结前加入同一明确边界处理，全流程正常结束。

完整产物位于`artifacts/recognition-small-anchor-v1/`与`artifacts/recognition-small-anchor-v2/`，每轮checkpoint、源码快照、真实图/灰图/同均值/错图预测、目标owner摘要、每step梯度与参数变化均保留。第二轮固定最终权重为`artifacts/recognition-small-anchor-v2/epoch-06/checkpoint.pt`；量化代理权重位于相邻`final_quantized/checkpoint.pt`，都不是固件。机器可读[汇总](../evidence/2026-11-27/small-anchor-training/summary.json)绑定设计、目标、训练、独立审计和checkpoint SHA。

训练后另外冻结并检查同一train32的小脸路径，无optimizer、oracle或新增validation/test：10个小脸的全部回归框覆盖从1/10升到9/10，但最终仍0/10。失败由原来的9个几何不足+1个置信度不足，变为1个几何不足、5个置信度不足、4个NMS/最终30框联合筛选丢失。不能把后4个单独归因于NMS，也不能用训练集路径诊断作为独立准确率。

预留100张test仍未进行模型推理。当前320/80数据均是人物+人脸正图，没有纯背景误报集，也不是相机域、低光、连续追焦或物理合焦验收。下一阶段优先检查按GT均衡的正例监督、难负例与候选排序，再考虑小脸尺度增强和原网络末端特征层的有限微调；每种改动应先验证变换/采样目标及量化范围。本轮不继续加轮或放宽验收阈值。

## K4权重布局修正（2026-11-27，未训练）

本节保留小尺度训练之前的原权重诊断状态。新发现的原权重排列按 `[O/4, (I×H×W)/4, O4, K4]` 存储，解码时先恢复完整 K 轴，再拆为 I/H/W。3×3的9个位置会跨越K4分块边界，因此旧的O/I/H/W六轴全排列未涵盖它。两份原模型多处完整零输入通道准确对应上游失活输出；conv1自然还原为此前HOW候选，1×1与原先受零填充约束支持的布局完全相同。

两个独立审查核对了共126个普通卷积的原字节、标量地址公式、零填充与往返；concat、slice、reshape、softmax合成检查通过。转置上采样不套用K4，bias顺序和缩放保持不变。这些约束与下面真实图行为共同支持修正，但没有原DSP逐层数值对照，仍不能宣称完整机内等价。

固定first32训练图、原始权重、未优化：人体AP50由36.35%升至74.61%，人脸由0.391%升至40.04%；K4两类平均真实图57.33%，同均值0.182%、灰117为0.284%、错图8.749%。随后冻结两种电脑输入几何策略，在原80张validation完成验证，未触碰预留100张test：

| 方案 | 人体AP50 | 人脸AP50 | 灰117人体／人脸 | 小脸命中 |
|---|---:|---:|---:|---:|
| 历史旧布局，拉伸 | 33.97% | 0.133% | 35.29%／0.068% | — |
| K4，拉伸384×256 | 60.59% | 34.33% | 0.848%／0.221% | 0/34 |
| K4，保持比例填边 | 63.20% | 38.91% | 1.025%／0.268% | 0/34 |
| 外部PicoDet-S320 | 88.41% | 不支持此类 | 30.04%／— | — |
| 外部BlazeFace原图 | — | 63.51% | —／0% | 25/34 |
| 外部BlazeFace384×256 | — | 73.59% | —／0.00053% | 19/34 |

AP为本地单IoU=0.5全点插值指标，不是COCO官方mAP。小脸统一按标注映射至384×256后sqrt(area)<32；命中为score≥0.5、IoU≥0.5、全类一对一匹配。K4的0/34不仅是置信度门槛问题：全部保存分数的最终候选也为0/34。拉伸方案人体TP70/FP16/FN50，人脸TP47/FP48/FN63；独立AP复算最大误差小于5e−16。保持比例方案只是电脑输入对照，未证实是原相机裁剪/缩放方法。

保持比例方案的错图对照在网络画布上用目标图的变换反映射，与外部模型的原图归一化错配不完全同义，因此不跨方案比较这类对照分数的细小差别。实际保持比例输入下会有42个小脸，正式表格始终使用共同参考下的34个。独立核验对保存预测可恢复的端点通过；clip丢失的原始端点不能从最终框完整还原。

六种固定训练集预处理对照支持减117：不减均值时平均AP降为40.11%；额外乘输入scale约0.0063时降为0.280%，与灰图相当；601/709灰度只产生小变化。用K4重做旧16种候选框分组/前景半区/类别头/anchor顺序对照，现有解码仍最好；原字节中两类proposal参数完全相同且共享bbox，未发现可将face单独改成另一组小anchors的证据。

另在固定train16做原图、四方向平移、目标遮挡和等面积背景遮挡，5个模型/方案共550次前向。旧S5人体条件跟随1/11，K4为27/31；K4人脸25/26，PicoDet人体38/42，BlazeFace两种输入的人脸51/51和42/42。这个分母只包括原先已命中的目标，不能遮掩漏检：K4所有80个人脸保留平移配对中只命中28个，小脸平移仍为0/34。可用脸部遮挡10对，原先有足够支持的仅2对；这2对目标遮挡相对背景遮挡的额外支持下降均值0.623，样本有限。

小脸逐阶段检查进一步定位了问题：固定train32中的10个小脸，原anchors在IoU≥0.5下覆盖0个；忽略分类分数、查看全部9600个回归框时仅覆盖1个；置信度≥0.5时为0个。因此9个主要卡在回归后的几何覆盖，另1个卡在分类置信度，原尺寸小脸并非主要被NMS删掉。仅作诊断的标注中心2倍/4倍裁剪将全部框覆盖提高到9/10和10/10，但最终真正命中仅2/10和3/10，同均值灰图均0/10。该实验利用标注知道目标位置，同时改变尺寸和保留的源细节，**不是自动检测方法或机内解决方案**；它说明仅放大仍不能解决全部漏检。

**当前 `ready_to_resume_recognition_training=false`。** 权重排列修正与小脸通过分开验收；不把外部模型成功、几何anchor覆盖或人工居中裁剪当成原网络小脸修复。原camera输入、定点舍入/饱和、pool、upsample边界、实际proposal ABI仍有未验证项。无DSP对照不是禁止电脑实验的理由，但当前小脸实测失败已经足以继续暂停训练。后续可在保留原计算图/25个anchor通道数的前提下，单独验证小尺度anchors与共享bbox训练目标；那属于明确的新检测配置，需要重新训练与回写验证，不能冒充原参数解码修复。

### 使用修正后的原权重

在带Torch/NumPy/Pillow的原S5实验环境中运行，输出目录必须是新目录：

```sh
python infer_s5_k4.py --firmware /path/to/44_hm_d_nw_sng.bin \
  --image /path/to/photo.jpg --output artifacts/k4-photo-example
```

得到`predictions.json`与`preview.png`；score只过滤预览，JSON保留decoder全部输出。默认按存储像素转灰、拉伸、减117，不隐式旋转EXIF；`--geometry letterbox`为单独的保持比例电脑策略。这里没有训练或固件回写。

核心可显式使用`CandidateNetwork(data, 'gemm_k4')`或`Detector(data, 'gemm_k4')`，普通卷积pack/unpack也接受`gemm_k4`。旧布局常量和默认值保留以复现历史；旧`train_recognition.py`的CLI仍只遍历三种历史布局，**不得直接用旧入口重训，也不得加载v2训练权重冒充修正原权重**。后续训练需要独立冻结K4、输入几何、小脸候选框设计、损失/目标审计和数据验收。原始诊断位于忽略目录`artifacts/recognition-diagnosis-v3/`，可提交的摘要在[此处](../evidence/2026-11-27/recognition-diagnosis-v3/summary.json)。

## 2026-11-27：历史v2验证和重跑结论

**目标分配错误已修复，六轮重跑完成，但未达到继续扩大训练的条件。** 本次从原始恢复权重开始，只更新三个检测头，实测 480 次 AdamW 更新、1,920 次训练图访问；主干完全未变。训练工具 22 项测试、数据契约 12 项测试、诊断工具 12 项自检通过。完成后重新核对每轮样本与优化器步数、三个 baseline／六轮预测的 AP、checkpoint 与实际 state 身份、validation 选择及冻结层，均通过。

原错误来自分别分配两类分类目标后，又独立分配共享回归目标。训练集中有 92 个人体正 anchor 和 564 个人脸正 anchor，其回归目标与任一同类 GT 的 IoU 都低于 0.5。修复后 train 的 883 个 GT、validation 的 230 个 GT 全部拥有独占覆盖匹配；分类／bbox 所有者冲突、漏匹配、抽样丢失、预算超限均为 0。此前检查曾把错误规则当作预期行为，该测试已替换为人脸／人体分别指向正确框的独立数值及梯度检查。

下表均是同一 80 张 validation 的 AP50（百分比），不是独立 test 成绩：

| 权重与输入 | 人体 AP50 | 人脸 AP50 | 两类平均 AP50 |
|---|---:|---:|---:|
| 原始恢复权重，真实照片 | 33.97% | 0.133% | 17.05% |
| 原始恢复权重，每图均输入灰117 | 35.29% | 0.068% | 17.68% |
| 固定第6轮，真实照片 | 25.69% | 0.355% | 13.02% |
| 固定第6轮，每图均输入灰117 | 18.79% | 0.285% | 9.54% |

六轮真实照片平均 AP50 依次为 15.91%、10.00%、9.45%、12.15%、11.65%、13.02%，均低于原始 17.05%，所以 `best.pt` 保留 epoch 0；`last.pt` 是实际更新后的第六轮。最后一轮超过其灰图对照，但没有整体提升，人脸绝对表现仍很低；这不足以把当前恢复模型称为已验证的 S5 识别网络。训练损失从第1轮 1.688 降至第6轮 0.787，也不能替代识别效果验收。

新 100 张 test 已完成文件与去重 QA，但未用于模型推理或指标评估，留作后续冻结方案的独立验收。验收条件在 `artifacts/recognition-v2/pre-run-plan.json` 中于训练前固定，判定保存在 `acceptance.json`。可提交的小报告见 [v2 summary](../evidence/2026-11-27/recognition-v2/summary.json)。

下一步先核对预处理、特征幅值／权重布局及候选框解码。小脸几何也是已测到的限制：validation 中缩放后 `sqrt(area)<32` 的 34 张脸，没有一个原 anchor 达到 IoU 0.5；全部解码框忽略分类分数时可覆盖 75/110 张脸，但排序和 NMS 后实际只匹配 16/110。候选框覆盖与类别排序都需继续查证，不能据此归因成原机网络缺陷。

历史 v1（未修复目标冲突）同样跑了 480 次更新；旧 test 平均 AP50 原始为 20.94%，最后一轮为 18.58%，best 同样保留 epoch 0。它与 v2 的数据用途不同，不把旧 test 数字与上表 validation 横向当作改善证据。

## 继续训练前的诊断（2026-11-27）

**尚未验证出可恢复训练的识别修复。** 本轮固定原始权重，优化器更新为 0；v2 的新 100 张 test 仍未做模型推理。预处理先用排序后的 32 张 train 做限定对照，再冻结每种首层解释的两个方案检查 80 张 validation；其余排列／定点试验只用相同 train32，解码试验只用 train8。validation 已参与诊断，不将它再称为独立 test。没有把本轮候选解释合入训练用的核心推理代码。

### 预处理

对照包括灰度转换、减均值、输入尺度和等比例补边，并同时评估真实照片、每图同均值灰图、固定灰117及错配照片。固件中存储的是 `int16 +14976 × 1/128 = +117`；“执行时减117”仍是待核对的运行逻辑。另一个输入输出尺度 `0.006317331455647945` 不能与该均值尺度混用：额外乘它会显著削弱图像相对灰图的特征差异。

下表是冻结方案在相同 80 张 validation 上的两类平均 AP50（百分比），不据此继续挑选或调参：

| 首层解释与输入 | 真实照片 | 同均值灰图 | 固定灰117 | 错配照片 |
|---|---:|---:|---:|---:|
| 原首层，减117 | 17.05% | 16.96% | 17.68% | 16.13% |
| 原首层，709灰度、减117 | 16.97% | 16.85% | 17.68% | 16.12% |
| H/O/W 首层重排，减117 | 16.82% | 16.20% | 17.68% | 17.88% |
| H/O/W 首层重排，不减均值 | 18.06% | 16.55% | 18.11% | 18.19% |

训练子集上小幅改善的方案没有通过验证集的最强对照，不能称为预处理修复。“同均值灰图”保留各图平均亮度；“错配照片”把另一张照片产生的预测配给当前真值，用于检查数据中的位置先验。它们用于否定假改善，不能单独证明原机输入格式。

### 权重排列与数值范围

- 两个固件模型共 8 个分类头，在所枚举的 24 种 1×1 轴排列中，仅 `Ob/Ib/O/I` 同时保留输出填充零和全部 25 对背景／前景权重的强反相关；24 种输出 lane 排列中仅恒等排列通过。**24 种输入 lane 排列仍全部通过这些约束**，所以还不能唯一确定输入通道语义。
- 首层 H/O/W 重排能让核呈现较连贯的边缘结构，但图像对照不支持把它当成已恢复的正确排列。16 个有限运行候选没有恢复可靠的小脸区分能力。
- 两个模型各 5 层，枚举全部 720 种外层／内层轴排列，共 7,200 项静态统计。没有达到预先设定的跨层候选门槛；最优统计仍有 48 个完全同分排列。核平滑度不能区分这些输入／输出通道顺序。
- 上采样原始表连同零填充，可精确分解成 2×2 相位结构，重组后得到现有 4×4 双线性系数；实际通道与相位寻址仍未确定。
- 逐层 int16 裁剪、舍入和 ReLU 后 unsigned16 假设均未通过真实图对照。浮点解释器先与原候选网络在非均匀输入上逐元素对齐；这只核验诊断实现。检测评分仍用浮点 softmax，因此此次定点试验不是完整原机定点推理链。超过假设 int16 范围也不能直接称为真机溢出。

### 小脸候选框

576 字节 proposal 参数尾部含两张 128 项 uint16 表。用分段最小二乘的一次函数近似 `exp(x)`，可精确复现 128/128 项斜率和 112/112 项非负截距。负截距的字节编码可拟合，但 DSP 的索引、符号、溢出和范围缩减仍未知；部分斜率饱和，直接将表用于完整区间会失败。找到生成公式不等于恢复了 bbox 解码器。

有界扫描没有找到第二套已连接到人脸头的小尺度 anchors。16 种分类／框通道及 anchor 解码候选，在三个主干排列下得到 144 组真实图／灰图／错配照片结果；保存的预测指标已逐组重算通过。人脸 AP50 最高的一组仍明显输给它自己的灰图和错配对照，未采纳。

原 anchor 对 validation 中 34 个缩小后面积平方根小于 32 像素的人脸，IoU≥0.5 覆盖为 0；假设将整套 scales 从 `[3,6,9,16,32]` 替换成 `[1,2,4,8,16]`，最小正方形 anchor 为 16 像素，每位置仍为 25 个 anchors，几何覆盖达到 15/34。**这是修改模型设计的效果，并非证明原机缺少小脸 anchors**。该结论依赖当前整图缩到 384×256 的映射；原机是否先放大 ROI 尚未确定。在通道对应关系和 bbox 数值语义未确定时，直接换 anchors 会改变原检测头的含义，不能算对原模型的还原修复。

### 证据与下一步

本轮新增脚本与本地报告分别覆盖 `preprocessing_investigation.py`、`weight_layout_investigation.py`、`weight_outer_axis_probe.py`、`anchor_decode_investigation.py`、`arithmetic_investigation.py` 和 ARM 输入链调查。完整原始诊断保存在忽略目录 `artifacts/recognition-diagnosis/`；可提交的摘要由 `summarize_recognition_diagnosis.py` 汇总既有证据、核对文件摘要，并明确保留未验证事项，不启动推理或训练。

ARM 调查已越过一个先前未知的间接调用：原固件 scatter copy 将 `0x070879b8` 的 `0x54fa4` 字节搬到 `0x08a67000`，据此恢复 RAM 函数表槽 `0x08aa7f04` 的初值为 Thumb 指针 `0x067b6457`。目标函数通过 `0x067c14b6` 将原描述符指针写入 MMIO `0x7c01f070`，随后操作控制／状态寄存器。这里只证明了描述符提交边界，没有找到 ARM 逐像素预处理，也未把这条邻近提交路径完整连接到 CDNN 推理队列。此结论来自原固件字节与搬运公式；QEMU 的外设返回值没有被用作原机证据。上游描述符生产者和下游硬件语义仍需追踪。

继续检查 `03_program` 的精确直接调用，仅找到仍原样转交入口描述符的 wrapper。使用原 loader 表确认的加载基址检查 `05_postboot1`、`07_postboot3`，未找到目标的直接分支或指针常量引用；`20_osdover` 没有已确认的 ARM 代码加载基址，仅检查指针常量，不猜地址反汇编。这是有限范围的阴性结果，不排除间接注册、其他组件或硬件内部生产输入。

继续训练的首要障碍是输入／早期特征及检测头数值解释尚未可信。下一步优先沿原固件实际输入调用链补齐裁剪、缩放和像素格式；用可靠原执行输入与 conv1／首个 3×3 输出，或恢复的 XM4 实际寻址代码，区分剩余排列。少量中间张量可以定位第一处分歧，但不能代替全网原机等价验收。只有冻结修复在真实图片上超过同均值、固定灰和错配对照，并让人体／人脸分别通过检查后，才重启有限训练；独立 test 留到方案冻结后。

这份诊断结论与数据准备就绪分开记录：`ready_for_computer_recognition_experiment` 仅说明电脑实验输入条件，**`ready_to_resume_recognition_training=false`** 才是当前是否继续训练的结论。没有原机输出不妨碍继续静态调查，但当前电脑端结果已经足以否定“可以直接扩大训练”。

## PicoDet／BlazeFace 外部基线已验证（2026-11-27，未训练）

已实际运行官方 PicoDet-S320 与 BlazeFace-FPN-SSH，在同一 80 张 validation 上完成真实图、每图 RGB 同均值平面、固定灰117、循环错配照片四组对照。每个模型／输入方案均先冻结计划，实际执行 240 次预测；三组共 720 次验证预测，错配对照由真实预测按图像宽高归一化映射，不另推理。使用原始预训练权重，没有 optimizer 更新，也没有使用预留的 100 张 test。用户已明确：不能确认新模型在 S5 上运行，就继续原网络路线；因此这两个模型只作为电脑端外部参照。

| 模型／输入 | 评估类别 | 真实图 AP50 | 同均值平面 AP50 | 灰117 AP50 | 错图 AP50 | 小脸命中／总数 |
|---|---|---:|---:|---:|---:|---:|
| PicoDet-S，320×320 | 人体 | 88.41% | 30.56% | 30.04% | 17.79% | 不适用 |
| BlazeFace，原图尺寸 | 人脸 | 63.51% | 0% | 0% | 0.045% | 25/34（73.53%） |
| BlazeFace，384×256 | 人脸 | 73.59% | 0% | 0.00053% | 0.077% | 19/34（55.88%） |

这里的小脸统一指原图标注映射到 384×256 后 `sqrt(area)<32`，而非 WIDER Hard 或各模型输入下同样大小。命中按置信度≥0.5、IoU≥0.5，与全部同类 GT 完成一对一匹配后统计；不先删大脸再算指标。人体为 120 个 GT，人脸 110 个 GT。相同固定阈值下，PicoDet 为 TP91/FP6/FN29；BlazeFace 原图为 TP66/FP24/FN44，384×256 为 TP78/FP11/FN32。原图方案更善于检出这组小脸，但缩小方案整体 AP 更高；没有根据结果改选阈值或重训。

仍有明确的尺度限制：在统一384×256参考尺寸下，中脸为32≤sqrt(area)<96，大脸为sqrt(area)≥96。原图／缩小方案中脸分别命中41/48、39/48，大脸分别为 **0/28、20/28**。因此原图单尺度并不适合本批数据中的大脸，不能把外部必要筛查通过写成“所有尺度没有问题”。这些是保存预测的分层观察，不据此推断网络内部原因，也没有事后融合多尺度结果来替换原报告。

三组均通过预先设定的外部基线必要筛查：本类 AP50≥0.5，比每个对照至少高0.15；人脸另要求小脸召回≥0.5且命中至少10个。独立核验器不导入模型或原评估实现，重算 AP、TP/FP/FN、大小分层、错配坐标和源码／预测摘要，三组全部通过，AP 最大差异约 `3.33e-16`。23项初始评估语义检查及新增依赖排除检查共24项通过；模型适配器另与官方预处理逐值一致，框缩放检查最大误差约0.000057像素。机器可读结果见 [外部基线摘要](../evidence/2026-11-27/external-baselines/summary.json)，本地可视化见 `artifacts/external-baselines/preview.png`。

这些结果说明这套数据和评估链可以得到明显依赖真实图像的检测结果；它们不验证 S5 原模型的权重排列或定点语义。80张全是人物＋人脸正样本，缺少空背景误报评估；公开预训练集与这些照片的历史重合未排除。COCO、WIDER 与 Open Images 对类别、遮挡和框范围的定义有差异，本结果不是它们的官方榜单成绩，也不是相机域合焦或低光验证。

PicoDet 使用 [官方模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/README.md) 的含后处理推理包，普通 COCO 权重含 `person`，没有 `person_face`，见 [类别源码](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/ppdet/data/source/category.py)。因此本次分别比较支持类别，没有把缺失类别记成0分后混算两类平均。人脸权重来自 [官方 BlazeFace 模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/face_detection/README.md)，使用 release/2.9 提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5` 导出。

PicoDet 的 [输入配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/_base_/picodet_320_reader.yml) 是 3 通道输入及逐通道归一化；[检测配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/_base_/picodet_v2.yml) 使用 stride 8/16/32/64 多尺度特征，以离散分布的期望计算框的四边距离。它提供了更细的检测网格，但小脸效果仍需对应人脸权重和同数据验收。不可套用现有 S5 的单通道减117、25 anchors／格和浮点 bbox 解码适配器。

电脑端跑通、量化导出和机内部署是不同验收项。[官方全量化示例](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/picodet/FULL_QUANTIZATION.md) 列出的已验证平台不包含 S5 的 CEVA/CDNN 路径；现阶段没有证据证明可直接载入 S5。还需单独验证算子支持、内存和耗时、量化误差及机内输入输出接口，参数量接近不能替代这些检查。原厂还原路线继续保留；外部基线不改写本轮 `verified_recognition_fix=false` 的结论，也不消耗预留 test。

### 外部基线的运行与复现

环境独立于原 S5 的 Torch 环境：`.venv-paddle`，Python3.12/macOS arm64。依赖锁定在 `requirements-external-baselines.txt`。Paddle2.6.2能够导出BlazeFace，但两模型的 `predictor.run()` 都曾 SIGSEGV，移出沙箱与关闭IR优化也失败；保持导出文件不变，改用 Paddle3.3.1 并设置 `FLAGS_enable_pir_api=0 OMP_NUM_THREADS=1` 后全部通过。不要把旧运行时失败当作模型识别失败。

BlazeFace 导出时仅应用了 [数据集初始化补丁](../patches/paddledetection-blazeface-export-without-dataset.patch)：把4项WIDER验证数据文件存在性断言移到真正读取数据的入口，使不读取数据的模型导出可执行；没有改变网络、权重或算子。补丁、导出命令和原始权重／模型摘要保存在 `artifacts/external-baselines/models/blazeface/provenance.json`。PicoDet来源记录在相邻 `models/picodet/provenance.json`。

PicoDet包内YAML的NMS阈值与实际图不同：YAML写0.3/0.5，图内实际score0.025/NMS0.6；适配器直接执行导出图，记录差异而不据YAML另做过滤。BlazeFace图内score0.01/NMS0.3。AP包含图输出的所有本类框，但仍受图内阈值和topK限制，不能称为无阈值的完整候选评价。

在已准备官方模型和上述独立环境后，用全新输出目录运行：

```sh
export FLAGS_enable_pir_api=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv-paddle/bin/python evaluate_external_baselines.py --model picodet \
  --model-dir artifacts/external-baselines/models/picodet/picodet_s_320_coco_lcnet \
  --output artifacts/external-baselines/reproduction/picodet
.venv-paddle/bin/python evaluate_external_baselines.py --model blazeface \
  --model-dir artifacts/external-baselines/models/blazeface/export/blazeface_fpn_ssh_1000e \
  --output artifacts/external-baselines/reproduction/blazeface
.venv-paddle/bin/python evaluate_external_baselines.py --model blazeface_s5_size \
  --model-dir artifacts/external-baselines/models/blazeface/export/blazeface_fpn_ssh_1000e \
  --output artifacts/external-baselines/reproduction/blazeface_s5_size
.venv-paddle/bin/python verify_external_evaluation_semantics.py
# 用 verify_external_baselines.py --help 查看保存预测的独立复算参数。
# 用 visualize_external_baselines.py --help 查看可视化参数；它不重新推理。
```

每组输出先保存 `frozen-plan.json`，完成后包含所有预测、逐GT匹配、指标与筛查、模型／源码／数据摘要。正式结果位于 `artifacts/external-baselines/runs/`，已另存 `independent-verification.json`；拒绝覆盖已有运行。外部基线的 `ready_for_external_baseline_use=true` 与原网络的 `ready_to_resume_s5_training=false` 分开保留。本次未训练 PicoDet、BlazeFace 或 S5 网络。

## 实验设计

- 固定输入为 `[1,1,256,384]`，即原图 RGB 转灰度、双线性拉伸到 384×256、减 117、比例 1。数据排除非平凡旋转，不隐式旋转标注。该预处理是明确的电脑实验假设。
- 比较 `o4_i4_hw`、`hw_o4_i4`、`o4_hw_i4` 三种 3×3 排列，仅以 80 张 validation 的平均 AP50 选定候选。分支结构、ReLU、上采样、类别头映射均仍是候选解释。
- 全网 1,456,392 个参数中，只训练原尺寸的两个分类头和共享框头（89,800 个参数）。其余恢复出的主干参数冻结，不增加部署尺寸。没有声称全网完成微调。
- 沿用恢复的 25 个 anchors／位置、16 像素步长、16×24 网格。v2 分类和共享 bbox 使用同一个 GT 所有者；人物和人脸各做二分类，但同一正 anchor 只归属一个类别与一个框。
- 常规正样本取两类并集中的最大 IoU≥0.5 目标；并集最大 IoU<0.3 为背景，中间忽略。另用确定性的独占匹配给每个 GT 保留一个有重叠的 anchor；覆盖优先，不声称总 IoU 最优。无可用匹配会失败，不静默丢弃。
- 每类每图抽样 128 个、最多 32 个正 anchor，必须保留上述覆盖匹配；超过预算明确失败。共享 bbox 最多抽样 128 个，必须包含所有已抽到的分类正样本。低 IoU 的强制匹配单独记录，不把它计作几何覆盖已达标。
- 损失为两类 cross-entropy 加 bbox SmoothL1（beta=1/9）。使用原始 logits 做稳定交叉熵；前 25 个背景通道与后 25 个目标通道按 anchor 正确配对。
- 恢复的 inclusive anchors 右下角加 1 后统一成连续 xyxy，目标编码与解码互逆。bbox 尺寸指数裁剪到 [-8,8]；NMS=0.3，每类 pre-NMS 3000／post-NMS 30。AP 使用全部保留分数，不只挑高分框。
- AdamW，学习率 0.0001，weight decay 0.0001，梯度范数裁剪 10；单图前向、4 图梯度累积，6 epochs，共 480 次 optimizer 更新。种子 20261127，CPU 2 线程，PyTorch deterministic algorithms。
- validation 挑选 epoch，未训练的 epoch 0 也可获选。v2 先用 `--skip-test` 跑完六轮，之后固定 best 和最后一轮做真实照片／灰117对照，不用该对照改选 checkpoint。预先约定：best 必须来自训练后、真实图平均 AP50 超过原始基线与该权重的灰图对照、两类 AP50 分别超过灰图，才进入新 test 验收；这些只是必要筛查条件，不是原机效果证明。

指标是 IoU=0.5 的全点插值 AP；不是 COCO mAP，也不是 Open Images 官方挑战成绩。AP50 不等于某张图片识别正确的概率。

## 数据与来源

数据来源、许可、人物子类别映射和排除规则详见 [数据说明](RECOGNITION_DATA.md)。首轮为 320 张训练、80 张验证、100 张独立测试；训练和验证取自官方 validation，测试取自官方 test。按摄影作者来源分组，同时检查源 ID、文件 SHA、解码 RGB 和 dHash 近重复，最多每位作者两张。

所有划分都要求有真实人物和人脸标注。它是有明确选择限制的公开图片小样本，未评估无人物场景的整图误报；没有低光真值、相机域内数据、连续追踪或合焦数据。原厂私有训练集未知，无法排除与公开图片历史重合；“独立”指相对本次微调。

数据 QA 发现不同 person 子类别可对同一人给出略不同框。筛选器按预先声明的跨子类重叠规则排除整图，避免重复真值；这也会保守排除真实重叠人群。旧数据试跑已停止于测试评估前，保留在本地 `artifacts/recognition-v1-superseded-label-duplicates/`。其后 v1 的旧 test 已用于诊断，不能再称为 v2 的独立验收。v2 保留原 320 train／80 validation，并新增 100 张 official test，排除旧 500 张的 ID、作者、文件／像素摘要及 dHash 近重复；当前最小 dHash 距离 10，排除阈值≤4。

## 复现命令

在 `af-research/` 中执行，使用安装了 `requirements.txt` 的 Python 3.12 环境：

```sh
export S5_AF_FIRMWARE_ROOT=/path/to/original-components
python recognition_data.py --output private-data/recognition-v1 \
  --train 320 --validation 80 --test 100 --seed 20261127 --workers 5
python prepare_recognition_v2_data.py --previous private-data/recognition-v1 \
  --output private-data/recognition-v2
python pretraining_readiness.py \
  --recognition-dataset private-data/recognition-v2/manifest.json \
  --output artifacts/recognition-v2-readiness.json
python verify_recognition_readiness.py
python verify_recognition_training.py --firmware-root /path/to/original-components
python audit_recognition_targets.py --dataset private-data/recognition-v2/manifest.json \
  --output artifacts/recognition-v2-target-audit.json
python verify_recognition_preflight.py --dataset private-data/recognition-v2/manifest.json \
  --output artifacts/recognition-v2-baseline-controls

# 以下为已完成的 v2 历史运行命令；当前诊断未通过，不据此启动新训练。
# 当时在工具和匹配验证通过后，从原始权重开始，并暂不运行新 test。
python train_recognition.py --dataset private-data/recognition-v2/manifest.json \
  --output artifacts/recognition-v2 --epochs 6 --skip-test
python verify_recognition_run.py --dataset private-data/recognition-v2/manifest.json \
  --run artifacts/recognition-v2
python verify_recognition_preflight.py --dataset private-data/recognition-v2/manifest.json \
  --run artifacts/recognition-v2 --output artifacts/recognition-v2/best-validation-controls
# 最后一轮对照：--checkpoint artifacts/recognition-v2/last.pt --expected-sha <report.json中的last.pt SHA256>
```

这些命令使用新输出目录；不要覆盖已冻结的 run。严格复现已有数据时，应携带冻结的 `manifest.json` 和 `provenance.json`，用 `recognition_data.py --from-manifest` 按原 SHA 下载，不能把重新抽样当作同一批数据。恢复要求原输出目录的 `last.pt`，并校验数据、源模型、代码与训练参数摘要。checkpoint 保存完整模型、AdamW、epoch、step、PyTorch／NumPy 随机状态；完整 epoch 边界可恢复，崩溃后的未完成 epoch 日志另行保留。只有上述必要条件通过，才去掉 `--skip-test`，以相同 `--epochs 6 --resume artifacts/recognition-v2/last.pt` 做最终测试，不追加训练。

训练不会写固件；`best.pt` 和 `last.pt` 含恢复参数，只在可信本地产物上用 `torch.load`。不要加载来源不明的 checkpoint。

## 产物和核验

本地忽略目录 `artifacts/recognition-v2/` 保存预先冻结的 `pre-run-plan.json`、源码快照、配置、三排列 validation 基线、逐样本损失日志、每轮 validation、权重更新检查、完整 checkpoint 和运行审计。预测分别绑定原组件 SHA256 与实际模型 state 摘要；不再把原组件摘要误当成微调权重身份。`private-data/recognition-v2/` 包含原图、官方元数据、来源许可、冻结数据清单及 QA。旧 run 原样保留；源码快照与权重在本地，仓库只保留源码、说明和不含图片／权重的小报告。

训练工具测试使用人工样例核验坐标、logit、目标分配、AP、梯度和恢复一致性，不计作真实识别性能。readiness 的真实完整文件校验、训练输出和独立 test 指标另行保存。

电脑实验条件和原机条件单独报告：即使 `ready_for_computer_recognition_experiment=true`，原机数值一致性、相机预处理／解码、定点量化、模型封装加载和真实镜头反馈仍未解决，部署准备保持 false。原最小 anchor 约 48 像素，对缩小后的小脸可能不合适；这只是下一轮需要检验的解释，不能据本实验直接宣称原机小脸能力差。
