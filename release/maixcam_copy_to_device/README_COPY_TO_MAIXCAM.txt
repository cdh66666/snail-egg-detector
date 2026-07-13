MaixCam 福寿螺卵识别部署包

当前版本：
- 模型：snail_eggs_yolov8n_640x480.mud + snail_eggs_yolov8n_640x480.cvimodel
- 程序：main.py
- 摄像头：640x480
- 推理：full_frame，全画面单次推理
- 默认阈值：CONF_TH = 0.18
- 后处理：模型分数 + 粉色比例 + 红色排除 + 大小/形状过滤

云台跟踪默认关闭。只有同时把 main.py 顶部的 ENABLE_GIMBAL 和
ENABLE_GIMBAL_TRACKING 改为 True，程序才会输出舵机 PWM。预测框、丢失目标
和不稳定目标都不会驱动舵机；控制器每次最多移动 2 度，绝对范围固定为
45 到 135 度。当前接线为水平 A19/PWM7、俯仰 A18/PWM6。
紧急禁用：在设备上创建 /root/snail_egg/disable_gimbal。

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
  ssh root@$MAIX_IP "echo $APP_ID > /maixapp/auto_start.txt && sync && reboot"

重启后检查：

  ssh root@$MAIX_IP "cat /maixapp/auto_start.txt && ps | grep main.py"

安全提醒：
程序只输出视觉候选目标。接入激光或其他执行器前，必须使用低功率指示灯或断开激光测试，并配合物理使能、急停、遮光门禁联锁和人工确认流程。
