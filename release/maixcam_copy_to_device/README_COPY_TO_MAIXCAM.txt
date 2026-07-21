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
  自动规则：整个视野约 2 秒没有有效卵团后关闭辅助瞄准灯
  手动规则：网页可无目标开启低功率辅助灯用于校准；急停始终强制关闭

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
  设备上电默认启动开放热点：
  SSID：SnailEgg-MaixCAM
  无需密码（开放热点）
  地址：http://192.168.66.1:8000/?token=maixcam

公司 Wi-Fi 已连接时手动切换热点
  1. 手机或电脑先打开当前网页控制地址，并确认设备在线。
  2. 点击“切换到设备热点”，确认提示。
  3. 等待设备断开公司 Wi-Fi，在手机 Wi-Fi 列表连接 SnailEgg-MaixCAM，无需输入密码。
  4. 手机通常会自动弹出调试页面；没有弹出时打开 http://192.168.66.1:8000/?token=maixcam 。
  5. 切换热点只改变网络，不会解除急停、打开激光或移动云台。
  6. 这个切换过程不需要重启；只有手机从公司 Wi-Fi 切到设备热点时，原网页连接会短暂断开。
  注意：热点模式下要回到公司 Wi-Fi，当前固件会在下次启动时重新尝试已保存网络。

配置公司 Wi-Fi
  在热点网页填写公司 Wi-Fi 名称和密码，点击“保存并连接公司 WiFi”。
  连接成功后查看状态中的 network_ip，并使用 http://<network_ip>:8000/?token=maixcam 。
  USB 有线维护地址固定为 http://10.178.38.1:8000/?token=maixcam 。

现场识别参数
  网页末尾提供模型推理下限、新目标确认阈值、重叠合并阈值和最低粉色占比。
  参数在下一帧生效，可点击“恢复推荐参数”回到发布默认值。
  参数只影响检测候选和轨迹建立，不会改变云台限幅、急停或激光安全逻辑。

原始场景录制
  点击“开始录制原始画面”后，设备保存 320x224 JPEG 帧序列到：
  /root/snail_egg/recordings/session_<time>/
  这些帧在绘制框、编号、文字和屏幕按钮之前保存，不含任何叠加内容。
  再次点击按钮停止录制。长时间录制后请及时通过 SSH/USB 导出并清理。

屏幕应急联网入口
  摄像头画面右下角显示 WIFI 或 HOTSPOT 按钮。
  热点模式点击 WIFI：使用网页上一次保存的 Wi-Fi 名称和密码主动连接。
  Wi-Fi 模式点击 HOTSPOT：恢复开放热点和固定地址 192.168.66.1。
  Wi-Fi 连接失败时会自动回滚到设备热点，不会永久失联。

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
