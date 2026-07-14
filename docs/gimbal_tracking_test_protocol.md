# MaixCam 云台跟踪测试方法

## 原则

每次只改变一个参数，并使用相同图片、相同轨迹、相同速度和相同随机种子复测。不要一边换图片、一边改 PID、一边改变摄像头距离，否则无法判断改善来自哪里。

测试期间不连接激光或其他执行器。舵机必须使用独立、稳压且电流足够的 5V 电源，并与 MaixCam 共地；不要让两个舵机长期从 MaixCam 5V 引脚取电。机械角度固定为 `45°~135°`。

## 测试顺序

1. 舵机单通道测试：确认 A19/PWM7 是水平轴，A18/PWM6 是俯仰轴。
2. 静止目标：目标居中保持 10 秒，确认 ID 稳定且云台不持续抖动。
3. 水平正弦：只评价水平轴方向、速度和超调。
4. 垂直正弦：只评价俯仰轴方向、速度和超调。
5. 矩形轨迹：评价拐角后的恢复速度。
6. 平滑随机轨迹：评价真实连续跟踪和短暂漏检恢复。
7. 多图轮换：最后才开启，用于评价不同距离、光照和外观下的识别稳定性。

## PC 端测试目标

```powershell
python scripts\gimbal_target_runner.py `
  --pptx "C:\Users\admin\Downloads\福寿螺图片 (1).pptx" `
  --model models\snail_eggs_yolov8n_640x480.pt `
  --mode sequence `
  --seed 20260714 `
  --log runs\gimbal_target\target_sequence.csv
```

默认使用 `520x420` 置顶小窗，像桌面目标一样在屏幕内移动，不会进入全屏。`--model` 会先用 PC 模型裁出卵团附近区域，让小目标在窗口内更清楚；`--pet-width` 和 `--pet-height` 可以调整窗口大小。

默认阶段是：居中 10 秒、水平 20 秒、垂直 20 秒、矩形 24 秒，然后持续运行平滑随机轨迹。右键或 `Esc` 关闭，空格暂停，数字键 `1` 到 `5` 切换轨迹。只有无人值守实验才使用 `--fullscreen`。

显示器被摄像头二次拍摄后会出现反光、白平衡偏移和摩尔纹，因此屏幕测试不能代替实物测试。必要时可用 `--saturation`、`--contrast` 和 `--brightness` 补偿测试图，但一次只改一个值。

定时轮换图片可加 `--image-period 6`。它适合先筛选哪些图片经过显示器二次拍摄后仍能稳定识别。

## 设备端启用和急停

发布源码默认不输出 PWM。先使用干跑模式验证目标选择、卡尔曼滤波和控制指令，干跑不会初始化 PWM：

```bash
mkdir -p /root/snail_egg
touch /root/snail_egg/gimbal_dry_run
rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/disable_gimbal
```

显示器拍摄测试可临时降低模型阈值，不影响正式阈值：

```bash
touch /root/snail_egg/screen_tracking_test
```

完成屏幕测试后必须删除：

```bash
rm -f /root/snail_egg/screen_tracking_test
```

确认舵机独立供电、方向、急停和机械限位后，才开启真实跟踪：

```bash
mkdir -p /root/snail_egg
rm -f /root/snail_egg/disable_gimbal
rm -f /root/snail_egg/gimbal_dry_run
touch /root/snail_egg/enable_gimbal_tracking
```

立即禁止云台：

```bash
touch /root/snail_egg/disable_gimbal
rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run
```

禁用文件优先级最高，存在时程序不会初始化 PWM。

## 参数调整顺序

1. 方向错误：只翻转 `GIMBAL_PAN_SIGN` 或 `GIMBAL_TILT_SIGN`。
2. 跟随太慢：逐步提高对应轴 `KP`，每次增加约 10%。
3. 超调明显：先降低 `KP`，再少量提高 `KD`。
4. 中心抖动：先增大 `GIMBAL_DEADZONE_X/Y`，不要先加积分。
5. 舵机动作过猛：降低 `GIMBAL_MAX_STEP_DEG`。
6. 稳态偏差：前五项稳定后，才考虑极小的 `KI`，并保留积分限幅。

第一轮建议：

```python
GIMBAL_MAX_STEP_DEG = 0.5
GIMBAL_CONTROL_HZ = 5.0
GIMBAL_DEADZONE_X = 0.070
GIMBAL_DEADZONE_Y = 0.070
GIMBAL_PAN_KI = 0.0
GIMBAL_TILT_KI = 0.0
```

## 日志评价

设备日志会输出 `AIM,TRACK`、`AIM,HOLD` 和 `STAT`。下载日志后运行：

```powershell
python scripts\analyze_gimbal_log.py runs\gimbal_device\gimbal.log `
  --output runs\gimbal_device\summary.json
```

主要指标：

- `p95_radial_error`：95% 控制更新的归一化中心误差。
- `hold_reasons`：漏检、预测框、低质量目标导致的保持次数。
- `track_ids`：同一轮出现过的锁定 ID，数量过多说明发生换轨。
- `mean_fps`：确认加入跟踪控制后没有明显拖慢识别。
