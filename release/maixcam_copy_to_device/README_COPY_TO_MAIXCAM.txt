MaixCAM 福寿螺卵识别与二轴云台部署包

当前候选版本
  检测模型：YOLO11n v26，固定输入 320x224
  摄像头画面：可独立使用 640x480；模型输入尺寸不等于摄像头输出尺寸
  发现阈值：0.35
  锁定保持阈值：0.10，仅作为已点选轨迹的弱测量，不能创建或切换目标
  推理节奏：YOLO 每个主循环帧运行一次
  漏检托底：卡尔曼最多保留同一目标 3 帧，预测框不继续驱动云台
  目标选择：手机网页点击绿色检测框
  丢失策略：保持当前位置，不自动跳到相邻目标
  云台：A18/PWM6 与 A19/PWM7；俯仰 80..120 度，左右 60..120 度，默认 90 度
  红色辅助瞄准灯：GPIOA15；连续按两次打开，按一次关闭
  安全规则：整个视野约 2 秒没有有效卵团后关闭辅助瞄准灯

压缩包内容
  main.py
  web_control.py
  preview.py
  gimbal_servo_test.py
  root/models/snail_eggs_yolo11n_320x224_v26.mud
  root/models/snail_eggs_yolo11n_320x224_v26.cvimodel

设备路径
  /maixapp/apps/snail_egg/main.py
  /maixapp/apps/snail_egg/web_control.py
  /maixapp/apps/snail_egg/preview.py
  /root/models/snail_eggs_yolo11n_320x224_v26.mud
  /root/models/snail_eggs_yolo11n_320x224_v26.cvimodel

部署但不设置自启动
  mkdir -p /maixapp/apps/snail_egg /root/models
  将 Python 文件复制到 /maixapp/apps/snail_egg/
  将 MUD 和 CVIMODEL 复制到 /root/models/
  cd /maixapp/apps/snail_egg
  python main.py

设置自启动
  printf %s "snail_egg" > /maixapp/auto_start.txt
  sync
  reboot

取消自启动
  : > /maixapp/auto_start.txt
  sync

手机控制
  程序启动后，在同一网络的手机浏览器打开：
  http://<MaixCAM-IP>:8000/?token=maixcam

户外热点
  设备 15 秒内未连接已保存 Wi-Fi 时自动启动：
  SSID：SnailEgg-MaixCAM
  密码：snailcam2026
  地址：http://192.168.66.1:8000/?token=maixcam

紧急停止云台
  mkdir -p /root/snail_egg
  touch /root/snail_egg/disable_gimbal

恢复云台
  rm -f /root/snail_egg/disable_gimbal
  touch /root/snail_egg/enable_gimbal_tracking

禁用红色辅助瞄准灯
  touch /root/snail_egg/disable_aim_relay

说明
  app_id 只是自定义应用目录名，本包统一使用 snail_egg。
  普通 MaixCAM 没有板载 IMU；船体扰动补偿需要外接 IMU 或后续加入全局光流。
  高功率消杀激光必须由人工控制，并保留物理使能、急停和遮光措施。
