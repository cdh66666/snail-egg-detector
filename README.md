# 福寿螺卵实时识别与二轴云台辅助瞄准

本项目面向 Sipeed MaixCAM：用 YOLO11 检测福寿螺卵团，在屏幕或手机网页显示实时画面，可人工选定一个目标后驱动二轴云台缓慢、限幅地移动到目标附近。红色辅助瞄准灯只作为低功率瞄准提示：自动模式随识别结果开启，网页也可手动开关用于校准；急停始终强制关闭。高功率消杀执行器必须由人工确认和操作。

## 当前基线

- 设备运行时：MaixPy / MaixCAM。
- 检测模型：YOLO11n，单类别 `eggs`，MaixCAM 输入 `320x224`，摄像头采集与网页画面可独立配置。
- 当前发布模型：`v26_union`。它在同口径 1000 帧压力集和 1378 张外部负样本集上均优于 v6；v27 因独立相机召回回落且外部误检增加而淘汰。
- 实机推理：YOLO11 每个主循环帧运行一次；卡尔曼只在偶发漏检时保留同一目标最多 3 帧，预测框不驱动云台继续移动。
- 检测策略：全画面以 `0.35` 高阈值生成可点击轨迹；人工点选后，`0.10..0.35` 的弱检测只能作为已锁定轨迹的测量，并且必须通过卡尔曼创新量、IoU、尺寸和形状一致性关联。弱检测不能生成新轨迹或切换目标。
- 运动补偿：跟踪预测使用舵机轨迹线程的实际 pan/tilt 输出作为控制输入，同时补偿 YOLO 双缓冲的一帧测量延迟；普通 MaixCAM 无板载 IMU，船体扰动需外接 IMU 或加入全局光流。
- 云台控制：独立 50 Hz 线程；俯仰 `80..120 deg`，左右 `60..120 deg`，默认 `90 deg`；两轴按同一二维方向向量固定小步插值，不做“先横后竖”，检测中心先低通再进入控制器。
- 跟踪原则：人工选定后保持锁定，不因短时漏检自动跳到相邻目标；丢失时保持当前位置，等待重新选定或人工微调。
- 网络视频：只保留最新 JPEG，网络慢时丢弃旧帧，避免手机端积压造成越来越大的延迟。
- 网络：设备上电默认启动开放热点 `SnailEgg-MaixCAM`，地址固定为 `http://192.168.66.1:8000/?token=maixcam`。热点守护线程持续检查无线接口、AP 和 DHCP，异常时自动重建；公司 Wi-Fi 连接失败也会自动回退热点。屏幕右下角保留 `WIFI/HOTSPOT` 触摸入口；USB 有线维护地址固定为 `http://10.178.38.1:8000/?token=maixcam`。
- 现场调参：网页末尾可分别调整模型推理下限、新目标确认阈值、重叠合并阈值和最低粉色占比，并可一键恢复推荐值。
- 原始数据录制：网页可录制模型输入尺寸的 `320x224` JPEG 帧序列到 `/root/snail_egg/recordings/session_<time>/`。帧在任何框、编号、文字和屏幕按钮绘制之前编码，可直接用于真实场景复核和后续训练。

## 评判标准

统一标准见 [`docs/acceptance_criteria.md`](docs/acceptance_criteria.md)。重要指标包括：

| 类别 | 目标 |
| --- | --- |
| 独立测试集召回率 | >= 92% |
| 独立测试集精确率 | >= 95% |
| 运动画面连续检测率 | >= 90% |
| 困难负样本误检 | <= 1 / 1000 帧 |
| 控制循环 | >= 25 FPS |
| YOLO 实际刷新 | >= 15 Hz |
| 云台 PWM | 50 Hz，P95 间隔 18--25 ms |
| 目标锁定 | 不自动换目标 |
| 瞄准灯安全 | 自动模式约 2 秒无有效卵团后关闭；急停在所有模式下强制关闭 |

这些指标必须用独立图片、连续视频和真实设备日志评估，不能用训练图或几张截图代替。

## 目录

```text
maixcam/main.py                         MaixCAM 主程序
maixcam/web_control.py                  手机网页控制
models/                                  PC 端训练/评估模型
data/                                    YOLO 数据集
scripts/yolo_train.py                   训练脚本
scripts/evaluate_thresholds.py          精度与阈值评估
scripts/evaluate_negative_fps.py        困难负样本误检评估
scripts/evaluate_camera_domain_holdout.py 独立 MaixCAM 域评测
scripts/audit_real_device_corpus.py    历史实机帧响应/连续性审计
scripts/test_gimbal_trajectory.py       二维轨迹与限幅测试
scripts/test_latest_frame_streamer.py   网络视频无积压测试
docs/acceptance_criteria.md             验收标准
docs/model_review_20260719.md           本轮模型和评测审计结论
release/maixcam_copy_to_device/         可上传到设备的文件
```

## PC 端复现

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/yolo_train.py --help
```

训练参数中的 `HEIGHT,WIDTH` 用于声明期望的导出形状；Ultralytics 训练阶段实际使用长边作为 `imgsz`，可配合 `--rect` 保留矩形 batch。脚本会在输出摘要中同时写出 `requested_imgsz`、`effective_train_imgsz` 和 `fixed_export_imgsz`，不再把训练阶段误写成固定非方形输入。

独立 MaixCAM 域评测（以下示例使用未用于训练的 v9 验证/留出部分）：

```powershell
python scripts\evaluate_camera_domain_holdout.py `
  --model runs\detect\runs_yolo\snail_eggs_yolo11n_union_v26-2\weights\best.pt `
  --camera runs\camera_domain_v9 `
  --report outputs\camera_v9_v26_review.json
```

报告中的 `annotated_box_recall` 是 IoU>=0.5 的标注框召回；`holdout_frame_detection_rate` 是没有框标注时的帧级命中，不能互相替代。历史截图审计同样只报告响应率和连续性代理指标。

训练和评估必须使用未参与训练的 `test` 集；困难负样本用空标签保存。示例：

```bash
python scripts/evaluate_negative_fps.py \
  --model runs/detect/runs_yolo/snail_eggs_yolo11n_union_v26-2/weights/best.pt \
  --negative-dir data/hard_negatives_mixed_v2/images \
  --imgsz 320 --conf 0.15 --safe-filter
```

## MaixCAM 一次运行

先在设备上开启 SSH，并保证电脑与 MaixCAM 在同一局域网。Windows PowerShell：

```powershell
$env:MAIXCAM_HOST = "192.168.x.x"
$env:MAIXCAM_PASSWORD = "root"
python scripts\maix_remote.py probe
python scripts\maix_remote.py deploy --app-id snail_egg --skip-models
python scripts\maix_remote.py run
```

发布包位于 [`release/maixcam_copy_to_device`](release/maixcam_copy_to_device)。v26 模型使用独立版本号，避免与旧模型混淆：

```text
release/maixcam_copy_to_device/root/models/snail_eggs_yolo11n_320x224_v26.mud
release/maixcam_copy_to_device/root/models/snail_eggs_yolo11n_320x224_v26.cvimodel
```

部署 v26 时，`maixcam/main.py` 中的 `MODEL` 应为：

```python
MODEL = "/root/models/snail_eggs_yolo11n_320x224_v26.mud"
```

首次只运行、不设置自启动；确认画面、检测、目标锁定和云台限幅都通过后，再按需要安装自启动：

```powershell
python scripts\maix_remote.py install-autostart --app-id snail_egg --reboot
```

## 安全约束

1. 舵机必须使用独立 5 V 电源并与 MaixCAM 共地，禁止用 3.3 V 引脚给舵机供电。
2. 代码只控制低功率红色辅助瞄准灯；高功率激光和消杀执行器保持人工确认。
3. 自动模式未检测到有效卵团约 2 秒后关闭辅助灯；手动开启用于校准，急停仍会强制关闭。
4. 首次上电调试应使用 `gimbal_dry_run`，确认方向和限幅后再接入真实云台。
5. 设备实际部署必须记录连续日志：`STAT`、`EGG`、`AIM`、`DOT`，不要只看截图。

## 官方资料

- [MaixPy YOLO11 自定义模型](https://wiki.sipeed.com/maixpy/doc/zh/vision/customize_model_yolov8.html)
- [MaixPy JPEG Streaming](https://wiki.sipeed.com/maixpy/doc/zh/video/jpeg_streaming.html)
- [MaixPy WebRTC Streaming](https://wiki.sipeed.com/maixpy/doc/zh/video/webrtc_streaming.html)
- [Ultralytics Training](https://docs.ultralytics.com/modes/train/)

YOLO11 与 YOLOv8 在 MaixCAM 的训练和转换流程相近，官方示例支持 `320x224` 输入。JPEG 推流适合调试预览，但不应把网络发送速度当成检测 FPS；本项目因此分别统计采集循环、YOLO 刷新和手机收到的画面帧率。
