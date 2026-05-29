福寿螺卵识别 MaixCam 部署包

把本目录内容复制到 MaixCam：

1. main.py
   - MaixVision 手动运行：放到项目根目录，作为 main.py 运行。
   - 开机自启：放到 /maixapp/apps/<app_id>/main.py。

2. root/models/snail_eggs_yolov8n_320x320.mud
   复制到 /root/models/snail_eggs_yolov8n_320x320.mud。

3. root/models/snail_eggs_yolov8n_320x320.cvimodel
   复制到 /root/models/snail_eggs_yolov8n_320x320.cvimodel。

开机自启示例：

如果 app_id 是 cdh1_：

  /maixapp/apps/cdh1_/main.py
  /maixapp/auto_start.txt 内容写成：cdh1_

当前默认是召回优先模式：

  ROUND_ROBIN_TILES = False
  TILES_PER_FRAME = 6
  CONF_TH = 0.15

这会每帧扫描全部 320x320 分块，识别更全，但 MaixCam 上约 5 FPS。

如果想提速，可在 main.py 顶部改成：

  ROUND_ROBIN_TILES = True
  TILES_PER_FRAME = 2 或 3

提速会降低同一时刻能识别到的卵团数量。用于激光瞄准时，建议先保持默认召回优先模式，并用低功率指示灯验证坐标稳定性。

安全提醒：

程序只输出视觉候选目标。接入激光或其他执行器前，必须使用低功率指示灯或断开激光测试，并配合物理使能、急停、遮光/门禁联锁和人工确认流程。
