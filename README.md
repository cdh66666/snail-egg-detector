# MaixCam 福寿螺卵实时识别

基于 YOLOv8n 和 Sipeed MaixCam 的福寿螺卵团实时检测程序。项目默认使用 640x480 摄像头画面做全画面单次推理，在屏幕上标出紧贴目标的检测框、中心十字和从左上到右下排序的 ID。

当前版本面向低延迟瞄准验证：不使用 tile 轮询，不保留旧帧记忆框，避免同一位置反复残留识别框。

> 安全提醒：本项目只输出视觉候选目标和坐标。任何激光、机械臂、喷洒或消杀执行器都必须加入物理使能、急停、遮光/门禁联锁、低功率预瞄准验证和人工确认流程。首次调试请断开激光，或用低功率指示灯替代。

![检测流程](assets/diagrams/pipeline.svg)

## 效果示例

![测试样本 1](assets/results/test_samples_1.jpg)

![测试样本 2](assets/results/test_samples_2.jpg)

![测试样本 3](assets/results/test_samples_3.jpg)

## 当前版本

| 项目 | 当前值 |
| --- | --- |
| 设备 | Sipeed MaixCam / MaixPy |
| 模型 | YOLOv8n |
| 输入 | 640x480 |
| 推理方式 | 全画面单次推理 |
| MaixCam 模型 | `snail_eggs_yolov8n_640x480.cvimodel` + `snail_eggs_yolov8n_640x480.mud` |
| 默认程序 | `release/maixcam_copy_to_device/main.py` |
| 开机自启 | 写入 `/maixapp/auto_start.txt` |

关键参数在 [maixcam/main.py](maixcam/main.py) 顶部：

```python
MODEL = "/root/models/snail_eggs_yolov8n_640x480.mud"
CONF_TH = 0.18
MIN_MODEL_CONF = 0.18
MAX_BOX_AREA_RATIO = 0.32
MAX_BOX_SIDE_RATIO = 0.86
MIN_PINK_RATIO = 0.035
SPEED_PROFILE = "full_frame"
FRAME_W = 640
FRAME_H = 480
USE_TILED_INFERENCE = False
```

`CONF_TH=0.18` 是当前实机部署的召回优先设置。接入执行器前建议先把 `CONF_TH` 和 `MIN_MODEL_CONF` 提高到 `0.25` 或 `0.35`，用低功率指示灯验证坐标稳定性后再调整。

## 离线评估

当前上线权重：

```text
runs/detect/runs_yolo/pinkeggs_yolov8n_640x480_hardneg_v5/weights/best.pt
```

对应的可发布文件已经放在：

```text
models/snail_eggs_yolov8n_640x480.pt
models/snail_eggs_yolov8n_640x480.onnx
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.cvimodel
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.mud
```

| 评估项 | 设置 | 结果 |
| --- | --- | --- |
| YOLO test split | `imgsz=480,640`，`conf=0.18`，安全过滤 | TP 91 / FP 6 / FN 4，Recall 95.79%，Precision 93.81% |
| YOLO test split | `imgsz=480,640`，`conf=0.25`，安全过滤 | TP 90 / FP 4 / FN 5，Recall 94.74%，Precision 95.74% |
| 混合负样本 holdout | 5183 张 COCO + Bing/Wikimedia/混杂负样本，`conf=0.18`，安全过滤 | 3 张图误检，图像级误检率 0.058% |
| 混合负样本 holdout | 5183 张 COCO + Bing/Wikimedia/混杂负样本，`conf=0.25`，安全过滤 | 2 张图误检，图像级误检率 0.039% |

评估结果保存在：

```text
runs/eval_640x480_v5_revised_filter.json
runs/eval_640x480_v5_revised_neg_conf018.json
```

`runs/` 默认不提交到 Git；如果需要复核，请在本地重新运行评估脚本。

## 快速部署到 MaixCam

仓库已经包含可直接复制到设备的部署包：

```text
release/maixcam_copy_to_device/
release/maixcam_copy_to_device.zip
```

Mac / Linux 终端示例：

```bash
MAIX_IP=192.168.10.107
APP_ID=cdh1_

ssh root@$MAIX_IP "mkdir -p /maixapp/apps/$APP_ID /root/models"
scp release/maixcam_copy_to_device/main.py root@$MAIX_IP:/maixapp/apps/$APP_ID/main.py
scp release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.* root@$MAIX_IP:/root/models/
ssh root@$MAIX_IP "echo $APP_ID > /maixapp/auto_start.txt && sync && reboot"
```

重启后检查：

```bash
ssh root@$MAIX_IP "cat /maixapp/auto_start.txt && ps | grep main.py"
```

看到类似下面的进程即部署成功：

```text
python3 /maixapp/apps/cdh1_/main.py auto_start
```

## Windows / VSCode 自动部署

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

设置设备地址并部署：

```powershell
$env:MAIXCAM_HOST='192.168.10.107'
$env:MAIXCAM_PASSWORD='root'
$env:PYTHONPATH='.codex_tools\paramiko'
python scripts\maix_remote.py install-autostart --app-id cdh1_ --reboot
```

只更新 `main.py`、不重传模型：

```powershell
python scripts\maix_remote.py install-autostart --app-id cdh1_ --skip-models --reboot
```

更多 VSCode + SSH 流程见 [docs/vscode_maixcam_workflow.md](docs/vscode_maixcam_workflow.md)。

## PC / Mac 图片和视频检测

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

检测图片：

```bash
python scripts/yolo_detect_media.py path/to/image.jpg \
  --model models/snail_eggs_yolov8n_640x480.pt \
  --conf 0.18 \
  --safe-filter
```

检测视频：

```bash
python scripts/yolo_detect_media.py path/to/video.mp4 \
  --model models/snail_eggs_yolov8n_640x480.pt \
  --conf 0.18 \
  --safe-filter \
  --video-stride 1
```

输出默认保存到 `runs/yolo_media/`，包括标注图、标注视频和 JSON 坐标。

## 复现训练

训练数据默认不提交到 Git。复现训练时需要准备 YOLO 格式数据集：

```text
data/yolo_pinkeggs_hardneg_v5_640x480/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
  pinkeggs_hardneg.yaml
```

现场采集到新的光照、距离、角度样本后，可以用训练集增强脚本生成额外鲁棒性样本：

```bash
python scripts/augment_field_robustness.py \
  --base-root data/yolo_pinkeggs_hardneg_v5_640x480 \
  --output-root data/yolo_pinkeggs_hardneg_v6_field_640x480 \
  --overwrite
```

这个脚本只扩充训练集，默认不改验证集和测试集，方便保持评估相对公平。

训练当前 640x480 模型：

```bash
python scripts/yolo_train.py \
  --data data/yolo_pinkeggs_hardneg_v5_640x480/pinkeggs_hardneg.yaml \
  --base-model yolov8n.pt \
  --epochs 45 \
  --imgsz 640 \
  --export-imgsz 480,640 \
  --batch 12 \
  --device 0 \
  --workers 0 \
  --project runs_yolo \
  --name pinkeggs_yolov8n_640x480_hardneg_v5 \
  --patience 12 \
  --rect \
  --export-onnx
```

说明：

- `--imgsz 640` 用于训练增强。
- `--export-imgsz 480,640` 导出固定 640x480 ONNX，匹配 MaixCam 摄像头。
- 实际上线前必须重新跑阈值扫描和负样本误检评估。

评估阈值：

```bash
python scripts/evaluate_thresholds.py \
  --model runs/detect/runs_yolo/pinkeggs_yolov8n_640x480_hardneg_v5/weights/best.pt \
  --data-root data/yolo_pinkeggs_hardneg_v5_640x480 \
  --split test \
  --imgsz 480,640 \
  --safe-filter \
  --confs 0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35 \
  --output runs/eval_640x480_v5_revised_filter.json
```

评估负样本误检：

```bash
python scripts/evaluate_negative_fps.py \
  --model runs/detect/runs_yolo/pinkeggs_yolov8n_640x480_hardneg_v5/weights/best.pt \
  --negative-dir data/coco/val2017 \
  --negative-dir data/hard_negatives_bing_v2 \
  --negative-dir data/hard_negatives_mixed_v2 \
  --negative-dir data/hard_negatives_wikimedia_v2 \
  --imgsz 480,640 \
  --conf 0.18 \
  --device 0 \
  --safe-filter \
  --exclude-yolo-root data/yolo_pinkeggs_hardneg_v5_640x480 \
  --output runs/eval_640x480_v5_revised_neg_conf018.json
```

## 转换为 MaixCam 模型

先把 ONNX 和 MUD 准备到转换目录：

```bash
python scripts/prepare_maixcam_package.py \
  --model-name snail_eggs_yolov8n_640x480 \
  --onnx models/snail_eggs_yolov8n_640x480.onnx \
  --mud maixcam/snail_eggs_yolov8n_640x480.mud \
  --input-width 640 \
  --input-height 480 \
  --calibration-images 200
```

在 WSL / TPU-MLIR 环境中转换：

```bash
VALIDATE_TRANSFORM=0 bash scripts/wsl_convert_maixcam_snail_eggs.sh
```

转换成功后会更新：

```text
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.cvimodel
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.mud
release/maixcam_copy_to_device.zip
```

## 坐标输出

设备端串口/终端会输出状态行：

```text
STAT,<frame>,FPS,<fps>,RAW,<raw_count>,CAND,<candidate_count>,EGGS,<target_count>,TILE,<tile_info>
```

目标行：

```text
EGG,<id>,<cx>,<cy>,<x>,<y>,<w>,<h>,<score>,<cx_norm>,<cy_norm>
```

字段说明：

- `id`：从左上到右下排序后的目标编号。
- `cx/cy`：检测框中心像素坐标。
- `x/y/w/h`：检测框左上角和宽高。
- `score`：模型置信度。
- `cx_norm/cy_norm`：0 到 1 的归一化中心点，便于下游控制器使用。

## 仓库结构

```text
assets/                         说明图和检测示例
docs/                           VSCode + SSH 工作流说明
maixcam/                        MaixCam 实时检测程序和 MUD 模板
models/                         PC 端权重和 ONNX 模型
release/maixcam_copy_to_device/ 可直接复制到 MaixCam 的部署包
scripts/                        采集、训练、评估、转换、部署脚本
requirements.txt                PC/Mac 端 Python 依赖
```

默认不提交的本地内容：

```text
data/
runs/
runs_yolo/
dist/
outputs/
*.cache
*.log
```

## 现场继续提升

模型效果最依赖真实现场数据。继续提升时建议按这个顺序补数据：

1. 用最终安装角度的 MaixCam 拍真实场景。
2. 把漏检卵团重新标注后加入正样本。
3. 把红色电机、线材、贴纸、手套、粉色玩具、反光物等误检对象加入负样本。
4. 重新训练、阈值扫描、负样本评估、转换、部署。
5. 接入执行器前先用低功率指示灯验证坐标稳定性。

## 参考

- [Sipeed MaixPy](https://github.com/sipeed/maixpy)
- [MaixPy 自定义 YOLOv8 模型文档](https://wiki.sipeed.com/maixpy/doc/zh/vision/customize_model_yolov8.html)
- [MaixCam ONNX 转 MUD 文档](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Pink-Eggs Dataset V1](https://datasetninja.com/pink-eggs-dataset-v1)
