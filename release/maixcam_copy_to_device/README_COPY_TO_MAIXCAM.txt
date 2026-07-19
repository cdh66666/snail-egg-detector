MaixCAM 福寿螺卵识别 v6 真机验证版

模型：YOLO11n v6
输入：320x224
主程序：main.py
模型描述：root/models/snail_eggs_yolo11n_320x224_v6.mud
模型文件：root/models/snail_eggs_yolo11n_320x224_v6.cvimodel

CVIMODEL SHA256：
B22411B77841B6F1B59924F5CA46EBAFAE96E2CA7798DEA78E7F1A56F7A44E66

设备路径：
/maixapp/apps/snail_egg/main.py
/maixapp/apps/snail_egg/web_control.py
/root/models/snail_eggs_yolo11n_320x224_v6.mud
/root/models/snail_eggs_yolo11n_320x224_v6.cvimodel

运行：
cd /maixapp/apps/snail_egg
python main.py

手机控制：
http://<MaixCAM-IP>:8000/

安全说明：程序只自动控制低功率红色辅助瞄准灯。高功率激光必须人工控制并配置物理急停。
