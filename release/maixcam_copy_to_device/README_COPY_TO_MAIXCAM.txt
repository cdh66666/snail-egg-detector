MaixCam 福寿螺卵识别部署包

当前默认版本：
- 模型：snail_eggs_yolov8n_640x480.mud + snail_eggs_yolov8n_640x480.cvimodel
- 程序：main.py
- 运行模式：SPEED_PROFILE = "full_frame"
- 摄像头：640x480
- 特点：单次全画面推理，不使用 tile 轮询，不显示记忆残留框

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

开机自启：

如果 app_id 是 cdh1_，则：

  /maixapp/apps/cdh1_/main.py
  /maixapp/auto_start.txt 的内容写成：cdh1_

Mac 终端示例：

  MAIX_IP=192.168.10.107
  APP_ID=cdh1_
  ssh root@$MAIX_IP "mkdir -p /maixapp/apps/$APP_ID /root/models"
  scp main.py root@$MAIX_IP:/maixapp/apps/$APP_ID/main.py
  scp root/models/snail_eggs_yolov8n_640x480.* root@$MAIX_IP:/root/models/
  ssh root@$MAIX_IP "echo $APP_ID > /maixapp/auto_start.txt && sync && reboot"

安全提醒：
程序只输出视觉候选目标。接入激光或其他执行器前，必须使用低功率指示灯或断开激光测试，并配合物理使能、急停、遮光/门禁联锁和人工确认流程。
