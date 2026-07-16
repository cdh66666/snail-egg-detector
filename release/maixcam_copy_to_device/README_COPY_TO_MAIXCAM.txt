MaixCam 福寿螺卵识别与云台跟踪部署包

文件：
  main.py
    -> /maixapp/apps/<app_id>/main.py

  root/models/snail_eggs_yolov8n_640x480.mud
  root/models/snail_eggs_yolov8n_640x480.cvimodel
    -> /root/models/

当前配置：
  摄像头/模型输入：640x480
  推理：full_frame 全画面单次推理
  模型阈值：0.20，附加粉色、红色、尺寸和形状过滤
  云台：A18/PWM6 水平，60-120 度
        A19/PWM7 俯仰，80-120 度
  红色瞄准灯：A15/GPIOA15；开按两次，关按一次
  1 m 参考点：(324,270)，对应右偏 10 mm、下偏 60 mm 的固定安装

部署但不设置自启动：
  mkdir -p /root/snail_egg /root/models
  将 main.py 和模型复制到上面的目标路径后，在应用目录运行：
  python main.py

设置自启动：
  APP_ID=snail_egg
  mkdir -p /maixapp/apps/$APP_ID
  cp main.py /maixapp/apps/$APP_ID/main.py
  printf %s "$APP_ID" > /maixapp/auto_start.txt
  sync
  reboot

说明：app_id 是自定义应用目录名，不是固定值。

云台默认不动作。先干跑：
  mkdir -p /root/snail_egg
  touch /root/snail_egg/enable_gimbal_tracking
  touch /root/snail_egg/gimbal_dry_run

确认方向、供电和限幅后启用真实输出：
  rm -f /root/snail_egg/gimbal_dry_run /root/snail_egg/disable_gimbal
  touch /root/snail_egg/enable_gimbal_tracking

紧急禁用：
  touch /root/snail_egg/disable_gimbal
  rm -f /root/snail_egg/enable_gimbal_tracking /root/snail_egg/gimbal_dry_run

约 0.35 m 显示器测试：
  touch /root/snail_egg/screen_tracking_test

恢复 1 m 室外参数：
  rm -f /root/snail_egg/screen_tracking_test

禁用红色瞄准灯：
  touch /root/snail_egg/disable_aim_relay

让下次启动重新执行一次红灯开关测试：
  rm -f /root/snail_egg/aim_relay_startup_test_done

安全要求：
  舵机使用独立稳压 5 V 电源并与 MaixCam 共地。高功率激光必须由人工控制，
  并配置物理使能、急停和遮光措施。首次调试必须断开高功率激光。
