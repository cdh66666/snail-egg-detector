# 福寿螺卵识别与二轴云台跟踪

这是部署在 Sipeed MaixCAM 上的福寿螺卵实时检测、手机点选和二轴云台辅助瞄准程序。

## 当前发布基线

GitHub `main` 只发布经过真机测试的 **YOLO11n v6**。后续模型仍在离线审查和训练，未经完整真机验收不得替换本基线。

| 项目 | 当前配置 |
| --- | --- |
| 模型 | YOLO11n v6，单类别 `eggs` |
| 模型输入 | `320x224` |
| 摄像头画面 | `640x480`，与模型输入独立 |
| 模型文件 | `snail_eggs_yolo11n_320x224_v6.cvimodel` |
| 推理 | MaixCAM NPU，双缓冲，检测步长 2 |
| 云台范围 | 左右 `60..120 deg`，俯仰 `80..120 deg`，中心 `90 deg` |
| 云台引脚 | A18/PWM6 与 A19/PWM7 |
| 辅助瞄准灯 | GPIOA15，仅低功率红色瞄准灯 |
| 高功率激光 | 必须人工控制并保留物理急停 |

v6 CVIMODEL SHA256：

```text
B22411B77841B6F1B59924F5CA46EBAFAE96E2CA7798DEA78E7F1A56F7A44E66
```

## 目录

```text
maixcam/main.py                         真机主程序
maixcam/web_control.py                  手机网页控制
release/maixcam_copy_to_device/         当前可部署文件
release/maixcam_copy_to_device.zip      当前可部署压缩包
scripts/                                训练、评测、部署与回归测试
docs/                                   设计与评测记录
known_good/Peter.1_20260718/             更早的稳定快照
```

## 部署

先在 MaixCAM 设置中开启 SSH，并让电脑和 MaixCAM 位于同一网络。

```powershell
$env:MAIXCAM_HOST = "192.168.x.x"
$env:MAIXCAM_PASSWORD = "root"
python scripts\maix_remote.py probe
python scripts\maix_remote.py deploy --app-id snail_egg
python scripts\maix_remote.py run
```

设备关键路径：

```text
/maixapp/apps/snail_egg/main.py
/maixapp/apps/snail_egg/web_control.py
/root/models/snail_eggs_yolo11n_320x224_v6.mud
/root/models/snail_eggs_yolo11n_320x224_v6.cvimodel
```

设置自启动：

```sh
printf %s "snail_egg" > /maixapp/auto_start.txt
sync
reboot
```

取消自启动：

```sh
: > /maixapp/auto_start.txt
sync
```

手机与 MaixCAM 在同一网络时打开：

```text
http://<MaixCAM-IP>:8000/
```

## 安全

- 程序只允许自动控制低功率红色辅助瞄准灯。
- 高功率消杀激光必须人工确认，配置物理使能、急停和遮光结构。
- 首次调试舵机时断开高功率激光。
- 舵机使用独立稳压 5 V 电源，并与 MaixCAM 共地。
- 摄像头、模型、程序或通信异常时应立即关闭辅助瞄准灯并保持云台。

## 候选模型规则

候选模型必须同时完成以下检查后才能替换 v6：

1. 未参与训练的真实 MaixCAM 标注集检测评测。
2. 静止、运动、暗光、偏角、距离变化和多目标连续视频评测。
3. 红色、粉色及复杂生活物体困难负样本误检评测。
4. MaixCAM 真机连续运行日志分析，包括检测频率、视频流频率和目标 ID 稳定性。
5. 云台跟踪期间不得自动切换目标、突变或越过机械限幅。

离线 PC 压力测试只能筛选候选，不能代替真机验收。

## 参考

- [MaixCAM 产品文档](https://wiki.sipeed.com/hardware/zh/maixcam/index.html)
- [MaixPy 文档](https://wiki.sipeed.com/maixpy/)
- [MaixPy 自定义 YOLO](https://wiki.sipeed.com/maixpy/doc/zh/vision/customize_model_yolov8.html)
- [MaixCAM 模型转换](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html)
