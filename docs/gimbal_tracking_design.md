# 二轴云台稳定跟踪设计

## 目标

让 MaixCam 在检测到多个目标时稳定锁住一个目标，并让云台平滑地把目标中心移动到画面中心。识别器、跟踪器和舵机控制器分开处理：

```text
YOLO 检测 -> 目标过滤 -> 多目标跟踪 -> 主目标锁定 -> 云台控制
```

默认只显示结果，不输出舵机 PWM。要进行真实舵机测试，必须同时打开：

```python
ENABLE_GIMBAL = True
ENABLE_GIMBAL_TRACKING = True
```

并确认舵机供电、机械结构和紧急断电方式已经准备好。

## 研究结论

### SORT

[abewley/sort](https://github.com/abewley/sort) 用卡尔曼滤波预测目标状态，再用数据关联把下一帧检测框分配给已有轨迹。它很轻，适合说明基本结构，但对长时间遮挡和重新进入画面处理较弱。

### ByteTrack

[FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack) 的关键点是不要简单丢弃低置信度框，而是先用高置信度框建立轨迹，再用低置信度框补齐暂时遮挡。这对小目标和偶发漏检很有价值，但完整 ByteTrack 依赖检测器输出和更多数据关联逻辑，直接放进 MaixCam 会增加延时和内存压力。

### OC-SORT

[noahcao/OC_SORT](https://github.com/noahcao/OC_SORT) 强调观测驱动的运动更新，对非线性运动和遮挡比普通 SORT 更稳。它很适合在 PC 端做高质量视频跟踪基准，但完整实现不适合当前 MaixPy 的低延时单文件部署。

### 云台控制参考

[PyImageSearch pan-tilt tracking](https://pyimagesearch.com/2019/04/01/pan-tilt-face-tracking-with-a-raspberry-pi-and-opencv/) 和 [Ball-Balancing-PID-System](https://github.com/giusenso/Ball-Balancing-PID-System) 都把图像误差、PID 和执行器分层，并强调死区、PID 调参和目标丢失时停止动作。这些原则已经移植到本项目的轻量控制器中。

## 本项目采用的取舍

当前 MaixCam 版本采用：

1. YOLOv8n 负责识别福寿螺卵。
2. 颜色和几何过滤只作为辅助条件，不代替 YOLO 形状识别。
3. 每个轨迹使用四个一维常速度卡尔曼滤波器，分别滤波中心点 `x/y` 和框宽高。
4. 用预测框与新检测框的距离和 IoU 做轻量数据关联，保持稳定 ID。
5. 允许短暂漏检时继续显示预测框，但预测框不会驱动舵机。
6. 主目标默认锁住当前最高分的稳定轨迹；也可以设置 `LOCK_TARGET_ID` 锁定指定 ID。
7. 云台控制每秒最多 8 次，带图像中心死区、PID、单次变化限幅和绝对角度限幅。

这样可以保留 MaixCam 的实时性，不把完整 PC 跟踪框架和深度特征模型搬到设备端。

## 当前参数

在 `maixcam/main.py` 顶部调整：

```python
GIMBAL_CONTROL_HZ = 8.0
GIMBAL_DEADZONE_X = 0.055
GIMBAL_DEADZONE_Y = 0.055
GIMBAL_MAX_STEP_DEG = 2.0
GIMBAL_MAX_OFFSET_DEG = 45
GIMBAL_MIN_STABLE = 3
GIMBAL_MIN_SCORE = 0.28
```

舵机接线默认是：

```text
水平轴: A19 / PWM7
俯仰轴: A18 / PWM6
```

如果方向相反，只修改：

```python
GIMBAL_PAN_SIGN = -1.0
GIMBAL_TILT_SIGN = 1.0
```

不要修改硬限幅来扩大运动范围。

## 失锁策略

- 目标不够稳定：保持舵机当前位置。
- 只有预测框、没有当前帧检测：保持当前位置。
- 置信度不足或轨迹已 miss：保持当前位置。
- 目标重新出现：沿用原轨迹 ID，重新满足稳定帧数后再允许控制。
- 程序退出：尝试回中后关闭 PWM。

## 验证方法

先保持两个开关为 `False`，确认设备画面和检测框正常。之后只接舵机，不接任何危险执行器，打开两个开关，观察串口：

```text
GIMBAL_INIT_OK
GIMBAL_TRACK_OK,HZ,8.0,DEADZONE,0.055,MAX_STEP,2.0
AIM,TRACK,<id>,EX,<x_error>,EY,<y_error>,PAN,<angle>,TILT,<angle>
```

任何时刻发现动作异常，建立 `/root/snail_egg/disable_gimbal` 文件并重启应用；程序检测到该文件后不会初始化 PWM。

## 后续可升级方向

- 在 PC 端用 Ultralytics 的 ByteTrack/OC-SORT 对录制视频做对比基准。
- 将低置信度检测框作为“只用于关联、不直接显示”的候选，以吸收偶发漏检。
- 对目标锁定加入外观/形状描述，避免两个相近目标交换 ID。
- 对云台做实测标定，分别测量水平和俯仰轴的方向、死区、响应速度和机械回差。

