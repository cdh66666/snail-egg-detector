MaixCam 福寿螺卵识别部署包

当前版本：
- 模型：snail_eggs_yolov8n_640x480.mud + snail_eggs_yolov8n_640x480.cvimodel
- 程序：main.py
- 摄像头：640x480
- 推理：full_frame，全画面单次推理
- 默认阈值：CONF_TH = 0.38（v10 INT8 真机正负动态验收值）
- 后处理：模型分数 + 粉色比例 + 红色排除 + 大小/形状过滤

云台跟踪默认关闭。当前实测接线为水平 A18/PWM6、俯仰 A19/PWM7；控制频率
10 Hz，最大角速度 6 度/秒、最大角加速度 12 度/秒方，绝对范围固定为
45 到 135 度。预测框、丢失目标
和不稳定目标不会驱动舵机。两个舵机应使用独立稳压 5V 电源并与 MaixCam 共地。

1 m 固定瞄准：激光位于摄像头右 10 mm、下 60 mm，640x480 的目标参考点
为 (324,270)。程序不识别红点；改变距离、镜头、分辨率或安装位置必须重标定。
创建 screen_tracking_test 进行约 0.35 m 桌面测试时，使用实测点 (329,326)；
删除该标志后自动恢复 1 m 参数。
屏幕模式阈值为 0.08，并排除瞄准点 32 px 内不超过 52x56 px 的小亮斑框，
防止锁住激光红点本身。

先用干跑模式验证跟踪，不输出 PWM：
  mkdir -p /root/snail_egg
  touch /root/snail_egg/gimbal_dry_run

确认供电、方向、急停和机械限位后，才开启真实跟踪：
  rm -f /root/snail_egg/gimbal_dry_run /root/snail_egg/disable_gimbal
  touch /root/snail_egg/enable_gimbal_tracking

紧急禁用：
  touch /root/snail_egg/disable_gimbal
  rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run

复制位置：

1. main.py
   放到：
   /maixapp/apps/<app_id>/main.py

2. root/models/snail_eggs_yolov8n_640x480.mud
   放到：
   /root/models/snail_eggs_yolov8n_640x480.mud

3. root/models/snail_eggs_yolov8n_640x480.cvimodel
   放到：
   /root/models/snail_eggs_yolov8n_640x480.cvimodel

Mac 终端示例：

  MAIX_IP=192.168.10.107
  APP_ID=cdh1_
  ssh root@$MAIX_IP "mkdir -p /maixapp/apps/$APP_ID /root/models"
  scp main.py root@$MAIX_IP:/maixapp/apps/$APP_ID/main.py
  scp root/models/snail_eggs_yolov8n_640x480.* root@$MAIX_IP:/root/models/
  ssh root@$MAIX_IP "sync"

需要开机自启时再执行：

  ssh root@$MAIX_IP "echo $APP_ID > /maixapp/auto_start.txt && sync && reboot"

重启后检查：

  ssh root@$MAIX_IP "cat /maixapp/auto_start.txt && ps | grep main.py"

安全提醒：
程序只输出视觉候选目标。接入激光或其他执行器前，必须使用低功率指示灯或断开激光测试，并配合物理使能、急停、遮光门禁联锁和人工确认流程。
