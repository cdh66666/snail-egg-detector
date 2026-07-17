# MaixCam 闭环云台测试方法

## 安全与固定条件

每轮只改变一个参数，并固定图片、轨迹、速度、距离和随机种子。程序测试只使用低功率红色辅助瞄准光；高功率激光必须断开或由人工保持物理禁用。

舵机使用独立稳压 5 V 电源并与 MaixCam 共地。水平轴硬限幅为 `60–120°`，俯仰轴为 `80–120°`，默认均为 `90°`，不得通过调参扩大。

## 测试顺序

1. 单通道：运行 `maixcam/gimbal_servo_test.py`，确认 A18/PWM6 水平、A19/PWM7 俯仰。
2. 干跑：只看锁定 ID、红点坐标和控制误差，不输出 PWM。
3. 静止单目标：连续 10 秒，检查抖动和静态误差。
4. 水平、垂直轨迹：分别检查方向、速度、超调和机械限位。
5. 平滑随机轨迹：检查短时漏检后的恢复。
6. 多目标：检查主目标不因相邻目标出现而跳转。
7. 框内扫描：最后启用，确认红点到达后才切换扫描点，所有点均留在锁定框内。

## PC 端小窗目标

```powershell
python scripts\gimbal_target_runner.py `
  --pptx "C:\Users\admin\Downloads\福寿螺图片 (1).pptx" `
  --model models\snail_eggs_yolov8n_640x480.pt `
  --mode sequence --seed 20260714 `
  --log runs\gimbal_target\target_sequence.csv
```

默认是置顶小窗，不占满工作屏幕。显示器二次拍摄会引入反光、白平衡偏移和摩尔纹，因此只能用于可重复控制测试，不能代替实物卵团测试。

## 设备端开关

干跑：

```bash
mkdir -p /root/snail_egg
touch /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run
```

真实红点闭环：

```bash
rm -f /root/snail_egg/gimbal_dry_run /root/snail_egg/disable_gimbal
touch /root/snail_egg/enable_gimbal_tracking /root/snail_egg/enable_closed_loop_aim
```

框内扫描：

```bash
touch /root/snail_egg/enable_aim_scan
```

紧急停止：

```bash
touch /root/snail_egg/disable_gimbal
rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run \
      /root/snail_egg/enable_closed_loop_aim /root/snail_egg/enable_aim_scan
```

## 调参顺序

1. 方向错误：只翻转对应的 `GIMBAL_PAN_SIGN` 或 `GIMBAL_TILT_SIGN`。
2. 红点识别不稳：先检查曝光和 RGB 红色峰值日志，再调红点阈值。
3. 跟随太慢：每次约提高对应轴 `KP` 10%。
4. 超调：先降低 `KP`，再小幅提高 `KD`。
5. 中心抖动：小幅增加死区或降低最大速度，不先加入积分。
6. 动作过猛：降低 `GIMBAL_MAX_STEP_DEG`、最大速度或最大加速度。

当前基准参数：

```python
GIMBAL_CONTROL_HZ = 10.0
GIMBAL_DEADZONE_X = 0.018
GIMBAL_DEADZONE_Y = 0.022
GIMBAL_MAX_STEP_DEG = 0.5
GIMBAL_MAX_RATE_DEG_S = 5.0
GIMBAL_MAX_ACCEL_DEG_S2 = 10.0
```

## 验收标准

- 主目标短时漏检时舵机保持，不朝预测位置盲动。
- 相邻卵团出现时不随意换锁；检测器 ID 重建时可在原位置附近重新绑定。
- 原目标持续遮挡约 5 秒后，能够自动释放旧锁并选择下一团有效目标。
- 红点测量新鲜时闭环误差持续收敛，扫描点切换前必须进入 `7 px` 到达范围。
- 框内扫描点始终位于锁定目标框的安全内缩区域。
- 水平角始终在 `60–120°`，俯仰角始终在 `80–120°`。
- 正常显示帧率与关闭调试录帧时的基准相比无明显下降。

主机回归：

```powershell
python scripts\test_closed_loop_aim.py
python scripts\test_target_lock_persistence.py
python scripts\test_gimbal_control_dynamics.py
```

设备日志重点查看 `DOT`、`AIM,TRACK`、`AIM,HOLD`、`LOCK,REACQUIRE` 和 `STAT`。下载日志后还可运行：

```powershell
python scripts\analyze_gimbal_log.py runs\gimbal_device\gimbal.log `
  --output runs\gimbal_device\summary.json
```
