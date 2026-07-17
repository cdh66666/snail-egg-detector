# 二轴云台闭环跟踪设计

## 目标与边界

系统在多团福寿螺卵中锁定一个稳定主目标，用摄像头内可见的低功率红色辅助瞄准点形成闭环，并在目标框内执行受限九点扫描：

```text
YOLO 检测 -> 过滤与多目标跟踪 -> 主目标锁定
                                  |
红点检测 -> 图像误差 -> 限速 PD -> 二轴云台 -> 新画面反馈
```

程序不控制高功率激光。高功率执行器必须由人工确认，并保留物理使能和急停。

## 跟踪方法

项目借鉴了 [SORT](https://github.com/abewley/sort)、[ByteTrack](https://github.com/ifzhang/ByteTrack) 和 [OC-SORT](https://github.com/noahcao/OC_SORT) 的状态预测、低分框关联和观测驱动思想，但采用适合 MaixPy 单文件运行的轻量实现：

1. YOLOv8n 识别卵团，颜色与几何规则只做二次复核。
2. 每条轨迹用常速度卡尔曼状态平滑中心和框尺寸。
3. 用距离、IoU、宽高和长宽比进行数据关联。
4. 主目标优先选择面积较大、置信度高、靠近画面中心且不贴边的轨迹。
5. 发生短时漏检时保持当前位置，不用预测框驱动舵机。
6. 检测恢复后先尝试在原位置附近重新绑定，减少相邻卵团之间跳转。
7. 原目标连续约 5 秒无法重新绑定、且存在其他有效目标时，释放旧锁并选择下一团。

## 红点闭环

闭环模式通过 `/root/snail_egg/enable_closed_loop_aim` 启用。程序在锁定目标附近搜索红点，使用 RGB 红色峰值、局部对比度和最大跳变约束排除普通红色区域。红点测量新鲜时，控制误差为：

```text
ex = target_x - red_dot_x
ey = target_y - red_dot_y
```

红点暂时不可见时保持当前舵机角度；持续失效后回到经过标定的固定参考点搜索，不盲目继续移动。`/root/snail_egg/enable_aim_scan` 会在红点到达当前点并稳定后，才切换到目标框内下一个九点位置。

固定参考点只用于启动和降级，不再作为主要瞄准依据。显示器近距离测试可创建 `/root/snail_egg/screen_tracking_test`，正式约 1 m 工作时必须删除该文件。

## 当前控制参数

```python
GIMBAL_CONTROL_HZ = 10.0
GIMBAL_DEADZONE_X = 0.018
GIMBAL_DEADZONE_Y = 0.022
GIMBAL_MAX_STEP_DEG = 0.5
GIMBAL_MAX_RATE_DEG_S = 5.0
GIMBAL_MAX_ACCEL_DEG_S2 = 10.0
GIMBAL_PAN_MIN_DEG = 60
GIMBAL_PAN_MAX_DEG = 120
GIMBAL_TILT_MIN_DEG = 80
GIMBAL_TILT_MAX_DEG = 120
```

接线：A18/PWM6 为水平轴，A19/PWM7 为俯仰轴，A15/GPIOA15 为低功率红色瞄准灯按键继电器。舵机应使用独立稳压 5 V 供电，并与 MaixCam 共地。

## 启用顺序

先干跑观察日志：

```bash
mkdir -p /root/snail_egg
touch /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run
```

确认方向、供电和机械限位后启用真实闭环：

```bash
rm -f /root/snail_egg/gimbal_dry_run /root/snail_egg/disable_gimbal
touch /root/snail_egg/enable_gimbal_tracking /root/snail_egg/enable_closed_loop_aim
```

中心闭环稳定后再启用框内扫描：

```bash
touch /root/snail_egg/enable_aim_scan
```

紧急停止云台并关闭闭环功能：

```bash
touch /root/snail_egg/disable_gimbal
rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run \
      /root/snail_egg/enable_closed_loop_aim /root/snail_egg/enable_aim_scan
```

## 日志与验收

`DOT` 日志给出红点位置、新鲜度和漏检次数；`AIM,TRACK` 同时给出目标、红点、期望点、归一化误差和舵机偏移。主机端运行：

```bash
python scripts/test_closed_loop_aim.py
python scripts/test_target_lock_persistence.py
python scripts/test_gimbal_control_dynamics.py
```

当前真机基准中，正常显示约为 `5.35 FPS`，末 20 次闭环更新的平均归一化误差约为水平 `0.0015`、垂直 `0.0145`。录制 JPEG 调试帧会额外降低帧率，不代表正常显示性能。
