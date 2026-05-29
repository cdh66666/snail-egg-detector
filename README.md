# 福寿螺卵实时识别器 MaixCam 版

这个项目用 MaixCam 实时识别福寿螺卵团。程序会在摄像头画面中画出紧贴卵团的框、中心十字和从左上到右下排序的 ID，同时在日志里输出中心点坐标，方便后续接云台、机械臂或低功率瞄准验证系统。

> 安全提醒：本项目只提供视觉候选目标。任何激光消杀装置都必须有物理使能、急停、遮光/门禁联锁、看门狗和人工确认流程。首次测试请断开激光，或用低功率指示灯代替。

![检测流程](assets/diagrams/pipeline.svg)

## 项目状态

- 板端设备：Sipeed MaixCam / MaixPy。
- 检测模型：YOLOv8n，输入尺寸 320x320。
- 默认模式：准确率优先，640x480 摄像头画面按 320x320 分块全量扫描。
- 板端帧率：约 5 FPS，画面复杂时会波动。
- 后处理：模型候选框 + 几何过滤 + 粉色系过滤 + 红色干扰过滤 + 连续帧稳定过滤。
- 显示策略：只显示最终识别出来的福寿螺卵，不显示未通过过滤的候选框。
- 当前默认适合“先稳定识别和瞄准验证”，不是最高帧率模式。

效果图：

![测试样本 1](assets/results/test_samples_1.jpg)
![测试样本 2](assets/results/test_samples_2.jpg)
![测试样本 3](assets/results/test_samples_3.jpg)

## 仓库里有什么

```text
assets/                         说明图和效果图
docs/                           VSCode + SSH 开发说明
maixcam/                        MaixCam 实时检测程序
models/                         PC 端 YOLO 权重和 ONNX
release/maixcam_copy_to_device/ 可直接复制到 MaixCam 的部署包
scripts/                        数据下载、YOLO 导出、训练、转换、部署脚本
requirements.txt                PC 端 Python 依赖
```

已包含可直接使用的文件：

- `release/maixcam_copy_to_device/main.py`
- `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud`
- `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel`
- `release/maixcam_copy_to_device.zip`
- `models/snail_eggs_yolov8n_320.pt`
- `models/snail_eggs_yolov8n_320x320.onnx`

本机训练数据、临时输出、调试视频、`runs/`、`data/`、`dist/` 和实验模型默认不进 Git。别人 clone 后可以按下面流程重新生成。

## 路线 A：直接部署预转换模型

这是最快复现路线，不需要重新训练。

1. 克隆仓库。

```powershell
git clone <your-repo-url>
cd snail-egg-detector
```

2. 把部署包复制到 MaixCam。

需要放置的文件：

| 仓库文件 | MaixCam 位置 |
| --- | --- |
| `release/maixcam_copy_to_device/main.py` | MaixVision 项目 `main.py`，或 `/maixapp/apps/<app_id>/main.py` |
| `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud` | `/root/models/snail_eggs_yolov8n_320x320.mud` |
| `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel` | `/root/models/snail_eggs_yolov8n_320x320.cvimodel` |

也可以直接解压：

```text
release/maixcam_copy_to_device.zip
```

3. 用 MaixVision 运行 `main.py`。

模型最终路径必须是：

```text
/root/models/snail_eggs_yolov8n_320x320.mud
/root/models/snail_eggs_yolov8n_320x320.cvimodel
```

## 路线 B：VSCode + SSH 自动部署

MaixCam 开启 SSH 后，可以不用 MaixVision 手工复制。

1. 安装 PC 端依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. 设置 MaixCam 地址。

```powershell
$env:MAIXCAM_HOST='192.168.10.107'
$env:MAIXCAM_PASSWORD='root'
```

把 IP 换成你自己板子的地址。默认账号通常是 `root/root`。

3. 探测设备。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 probe
```

4. 安装为开机自启。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 install-autostart --app-id cdh1_ --reboot
```

这条命令会：

- 上传 `maixcam/main.py` 到 `/maixapp/apps/cdh1_/main.py`
- 上传模型到 `/root/models/`
- 写入 `/maixapp/auto_start.txt`
- 清除 `/root/snail_egg/headless`
- 重启后自动进入识别程序

调试时也可以只上传并运行 20 秒：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 deploy-run --skip-models --run-seconds 20
```

更多 VSCode 任务见 [docs/vscode_maixcam_workflow.md](docs/vscode_maixcam_workflow.md)。

## PC 端图片/视频检测

检测图片：

```powershell
python scripts\yolo_detect_media.py path\to\image.jpg --model models\snail_eggs_yolov8n_320.pt
```

检测视频：

```powershell
python scripts\yolo_detect_media.py path\to\video.mp4 --model models\snail_eggs_yolov8n_320.pt --video-stride 1
```

输出默认保存到 `runs/yolo_media/`，包括标注图/视频和 JSON 坐标。

## 从公开数据重新训练

这条路线用于复现训练过程，或继续改进模型。公开数据来自 Pink-Eggs Dataset V1，下载脚本会整理成项目中间格式。

1. 下载并整理数据。

```powershell
python scripts\collect_pinkeggs_dataset.py --source full --limit 960 --output-dir data\pinkeggs_full_960
```

2. 转成 YOLO 数据集。

```powershell
python scripts\export_yolo_labels.py --dataset data\pinkeggs_full_960\annotations.json --output-dir data\yolo_pinkeggs_full_960 --clean
```

生成结构：

```text
data/yolo_pinkeggs_full_960/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
  pinkeggs.yaml
```

3. 训练 YOLOv8n。

有 NVIDIA GPU：

```powershell
python scripts\yolo_train.py --data data\yolo_pinkeggs_full_960\pinkeggs.yaml --base-model yolov8n.pt --epochs 50 --imgsz 320 --batch 16 --device 0 --export-onnx
```

只有 CPU：

```powershell
python scripts\yolo_train.py --data data\yolo_pinkeggs_full_960\pinkeggs.yaml --base-model yolov8n.pt --epochs 20 --imgsz 320 --batch 8 --device cpu --export-onnx
```

训练输出在 `runs_yolo/`。把最好的 `.pt` 和导出的 `.onnx` 复制到 `models/` 后，可以继续走 MaixCam 转换。

## 转换到 MaixCam

MaixCam 不能直接运行 `.pt` 或普通 `.onnx`。转换链路是：

```text
YOLO .pt -> ONNX .onnx -> MaixCam .cvimodel + .mud
```

推荐在 WSL Ubuntu 22.04 中安装 TPU-MLIR 环境：

```powershell
wsl -d Ubuntu-22.04 -- bash scripts/wsl_install_tpu_mlir.sh
```

准备转换目录：

```powershell
python scripts\prepare_maixcam_package.py --calibration-images 200
```

执行转换并重建 release 包：

```powershell
wsl -d Ubuntu-22.04 -- bash scripts/wsl_convert_maixcam_snail_eggs.sh
```

成功后会生成：

```text
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud
release/maixcam_copy_to_device/main.py
release/maixcam_copy_to_device.zip
```

如果转换脚本提示缺少校准图片，请先完成“从公开数据重新训练”的第 1、2 步。

## 板端关键参数

参数在 `maixcam/main.py` 顶部。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `FRAME_W`, `FRAME_H` | `640`, `480` | 摄像头输入画面 |
| `USE_TILED_INFERENCE` | `True` | 用 320x320 分块扫大画面，提高小目标召回 |
| `ROUND_ROBIN_TILES` | `False` | 默认每帧扫全部分块，识别数量优先 |
| `TILES_PER_FRAME` | `6` | 轮询模式下每帧扫描的分块数 |
| `CONF_TH` | `0.05` | YOLO 原始候选阈值 |
| `MIN_MODEL_CONF` | `0.05` | 最低模型置信度 |
| `STRONG_MODEL_CONF` | `0.25` | 高置信目标阈值 |
| `ENABLE_COLOR_GATE` | `True` | 开启粉色/红色过滤 |
| `MIN_PINK_RATIO` | `0.18` | 检测框内部粉色比例阈值 |
| `MAX_RED_BAD_RATIO` | `0.55` | 红色干扰占比阈值 |
| `REQUIRE_STABLE_FRAMES` | `2` | 连续出现多少帧才显示 |
| `DUAL_BUFF` | `False` | 保持检测框和当前帧对齐，适合瞄准 |

性能档位：

| 目标 | 推荐设置 | 取舍 |
| --- | --- | --- |
| 识别数量优先 | `ROUND_ROBIN_TILES = False` | 约 5 FPS，召回最好 |
| 平衡模式 | `ROUND_ROBIN_TILES = True`, `TILES_PER_FRAME = 3` | 约 10 FPS，可能漏小目标 |
| 帧率优先 | `ROUND_ROBIN_TILES = True`, `TILES_PER_FRAME = 2` | 约 15 FPS，召回明显下降 |

## 坐标输出格式

状态行：

```text
STAT,<frame>,FPS,<fps>,RAW,<raw_count>,CAND,<candidate_count>,EGGS,<target_count>,TILE,<tile_info>
```

目标行：

```text
EGG,<id>,<cx>,<cy>,<x>,<y>,<w>,<h>,<score>,<cx_norm>,<cy_norm>
```

含义：

| 字段 | 说明 |
| --- | --- |
| `id` | 从左上到右下排序后的编号 |
| `cx`, `cy` | 检测框中心点像素坐标 |
| `x`, `y`, `w`, `h` | 检测框位置和尺寸 |
| `score` | 模型置信度 |
| `cx_norm`, `cy_norm` | 归一化中心点坐标，范围 0 到 1 |

## 继续提升现场效果

现场效果主要靠补真实数据：

1. 用最终安装角度的 MaixCam 拍摄真实场景。
2. 把漏检的卵团重新标注后加入正样本。
3. 把红色电机、线材、贴纸、手套、反光物等误检对象加入负样本。
4. 重新训练、转换、部署。
5. 接入激光前，先用低功率指示灯验证坐标稳定性。

## 参考资料

- [Sipeed MaixPy](https://github.com/sipeed/maixpy)
- [MaixPy YOLO 目标检测文档](https://wiki.sipeed.com/maixpy/doc/zh/vision/yolov5.html)
- [MaixPy 自定义 YOLOv8 模型文档](https://wiki.sipeed.com/maixpy/doc/zh/vision/customize_model_yolov8.html)
- [MaixCam ONNX 转 MUD 文档](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [Pink-Eggs Dataset V1](https://datasetninja.com/pink-eggs-dataset-v1)
