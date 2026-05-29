# 福寿螺卵实时识别器 MaixCam 版

这个项目用于在 Sipeed MaixCam 上实时识别福寿螺卵团，并在画面中标出紧贴目标的框、中心十字和从左上到右下排序的 ID。项目同时包含 PC 端图片/视频检测脚本、YOLO 训练脚本、难负样本挖掘脚本、MaixCam 转换脚本和 SSH 自动部署脚本。

> 安全提醒：本项目只输出视觉候选目标和坐标。任何激光、机械臂、喷洒或消杀执行器都必须加入物理使能、急停、遮光/门禁联锁、低功率预瞄准验证和人工确认流程。首次调试请断开激光，或用低功率指示灯替代。

![检测流程](assets/diagrams/pipeline.svg)

## 当前状态

- 设备：Sipeed MaixCam / MaixPy
- 模型：YOLOv8n，输入尺寸 320x320，已转换为 MaixCam `cvimodel + mud`
- 默认画面：640x480 摄像头输入，按 320x320 tile 全量扫描，优先保证识别数量
- 默认帧率：约 5 FPS，取决于画面复杂度和 tile 数量
- 后处理：YOLO 候选框 + 几何过滤 + 粉色/红色安全门 + 连续帧稳定过滤
- 显示：只显示最终通过过滤的卵团，不显示被过滤掉的候选框
- 开机自启：支持通过脚本写入 `/maixapp/auto_start.txt`

效果示例：

![测试样本 1](assets/results/test_samples_1.jpg)
![测试样本 2](assets/results/test_samples_2.jpg)
![测试样本 3](assets/results/test_samples_3.jpg)

## 验收指标

当前上线权重选用 `runs/detect/runs_yolo/pinkeggs_yolov8n_hardneg_v2/weights/best.pt` 导出的模型。后续又试过 v3/v4 追加训练，但独立负样本误检更差，所以没有上线。

固定测试结果：

| 评估项 | 设置 | 结果 |
| --- | --- | --- |
| YOLO test split | `conf=0.15`，带安全过滤 | TP 86 / FP 17 / FN 9，Recall 90.5%，Precision 83.5% |
| COCO 负样本 holdout | 3700 张未进入训练的 COCO 图，`conf=0.15`，带安全过滤 | 10 张图出现误检，图像级误检率 0.27% |
| COCO 负样本 holdout | 3700 张未进入训练的 COCO 图，`conf=0.35`，带安全过滤 | 3 张图出现误检，图像级误检率 0.081%，但召回会下降 |

默认部署采用召回优先的 `conf=0.15`。如果你要接入激光执行器，建议先用 `conf=0.35` 或更高做低功率预瞄准验证，再逐步降低阈值。

## 仓库结构

```text
assets/                         说明图和检测示例
docs/                           VSCode + SSH 工作流说明
maixcam/                        MaixCam 实时检测程序
models/                         当前 PC/转换用模型
release/maixcam_copy_to_device/ 可直接复制到 MaixCam 的部署包
scripts/                        采集、训练、评估、转换、部署脚本
requirements.txt                PC 端 Python 依赖
```

本地大数据和训练输出默认不进入 Git：`data/`、`runs/`、`dist/`、`*.cache`、实验 hardneg 权重等。

## 快速部署

已经生成好的部署文件在：

```text
release/maixcam_copy_to_device/main.py
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel
release/maixcam_copy_to_device.zip
```

手动复制到 MaixCam：

| 仓库文件 | MaixCam 位置 |
| --- | --- |
| `release/maixcam_copy_to_device/main.py` | `/maixapp/apps/<app_id>/main.py` 或 MaixVision 项目 `main.py` |
| `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud` | `/root/models/snail_eggs_yolov8n_320x320.mud` |
| `release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel` | `/root/models/snail_eggs_yolov8n_320x320.cvimodel` |

## VSCode + SSH 自动部署

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

设置 MaixCam 地址：

```powershell
$env:MAIXCAM_HOST='192.168.10.107'
$env:MAIXCAM_PASSWORD='root'
```

探测设备：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 probe
```

部署并设置开机自启：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 install-autostart --app-id cdh1_ --reboot
```

这条命令会上传程序和模型，写入 `/maixapp/auto_start.txt`，清除 headless 标记，然后重启设备。

更多细节见 [docs/vscode_maixcam_workflow.md](docs/vscode_maixcam_workflow.md)。

## PC 图片/视频检测

检测图片：

```powershell
python scripts\yolo_detect_media.py path\to\image.jpg --model models\snail_eggs_yolov8n_320.pt --conf 0.15 --safe-filter
```

检测视频：

```powershell
python scripts\yolo_detect_media.py path\to\video.mp4 --model models\snail_eggs_yolov8n_320.pt --conf 0.15 --safe-filter --video-stride 1
```

输出默认保存到 `runs/yolo_media/`，包括标注图片/视频和 JSON 坐标。

## 完整复现训练

1. 收集并导出正样本：

```powershell
python scripts\collect_pinkeggs_dataset.py --source full --limit 500 --output-dir data\pinkeggs_full_500
python scripts\export_yolo_labels.py --dataset data\pinkeggs_full_500\annotations.json --output-dir data\yolo_pinkeggs_clean_500 --clean
```

2. 准备负样本。推荐下载 COCO val2017 作为生活场景负样本池：

```powershell
New-Item -ItemType Directory -Force data\coco
curl.exe -L --fail --retry 3 -o data\coco\val2017.zip http://images.cocodataset.org/zips/val2017.zip
Expand-Archive data\coco\val2017.zip -DestinationPath data\coco -Force
```

也可以额外尝试公开网页难负样本采集，网络环境不稳定时会比较慢：

```powershell
python scripts\collect_bing_hard_negatives.py --output-dir data\hard_negatives_bing_v2
python scripts\collect_wikimedia_hard_negatives.py --output-dir data\hard_negatives_wikimedia_v2
```

3. 构建 hard-negative YOLO 数据集：

```powershell
python scripts\build_hard_negative_yolo.py --positive-root data\yolo_pinkeggs_clean_500 --negative-dir data\hard_negatives_mixed_v2\images --output-dir data\yolo_pinkeggs_hardneg_v2 --clean --seed 20260529
python scripts\augment_yolo_snail_data.py --data-root data\yolo_pinkeggs_hardneg_v2 --train-montage 220 --train-negatives 360 --val-montage 30 --val-negatives 70 --seed 20260529
```

如果你没有 `data\hard_negatives_mixed_v2\images`，可以先从 COCO 或自己拍摄的非卵图片中整理一个只包含背景/干扰物的目录，再传给 `--negative-dir`。

4. 训练并导出 ONNX：

```powershell
python scripts\yolo_train.py --data data\yolo_pinkeggs_hardneg_v2\pinkeggs_hardneg.yaml --base-model yolov8n.pt --epochs 70 --imgsz 320 --batch 32 --device 0 --workers 0 --project runs_yolo --name pinkeggs_yolov8n_hardneg_v2 --patience 16 --export-onnx
```

5. 扫阈值和误检率：

```powershell
python scripts\evaluate_thresholds.py --model runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.pt --data-root data\yolo_pinkeggs_hardneg_v2 --split test --imgsz 320 --safe-filter --confs 0.15,0.20,0.25,0.30,0.35,0.40 --output runs\eval_hardneg_v2_safe.json
python scripts\evaluate_negative_fps.py --model runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.pt --negative-dir data\coco\val2017 --imgsz 320 --conf 0.15 --safe-filter --exclude-yolo-root data\yolo_pinkeggs_hardneg_v2 --output runs\eval_hardneg_v2_coco_holdout.json
```

6. 可选：挖误检并复训。只有当复训后的独立负样本误检率和 test recall 都更好时，才建议替换上线模型。

```powershell
python scripts\mine_false_positives.py --model runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.pt --base-data-root data\yolo_pinkeggs_hardneg_v2 --output-data-root data\yolo_pinkeggs_hardneg_v3 --negative-dir data\coco\val2017 --imgsz 320 --conf 0.05 --device 0 --clean
python scripts\yolo_train.py --data data\yolo_pinkeggs_hardneg_v3\pinkeggs_hardneg.yaml --base-model runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.pt --epochs 50 --imgsz 320 --batch 32 --device 0 --workers 0 --project runs_yolo --name pinkeggs_yolov8n_hardneg_v3 --patience 14 --export-onnx
```

## 转换到 MaixCam

MaixCam 不能直接运行 `.pt` 或普通 `.onnx`。转换链路是：

```text
YOLO .pt -> ONNX .onnx -> MaixCam .cvimodel + .mud
```

准备转换目录：

```powershell
Copy-Item runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.pt models\snail_eggs_yolov8n_320.pt -Force
Copy-Item runs\detect\runs_yolo\pinkeggs_yolov8n_hardneg_v2\weights\best.onnx models\snail_eggs_yolov8n_320x320.onnx -Force
python scripts\prepare_maixcam_package.py --calibration-images 200
```

在 WSL Ubuntu 22.04 的 TPU-MLIR 环境里转换并重建 release 包：

```powershell
wsl.exe -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/admin/Documents/GitHub/snail-egg-detector && bash scripts/wsl_convert_maixcam_snail_eggs.sh"
```

成功后会更新：

```text
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.cvimodel
release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_320x320.mud
release/maixcam_copy_to_device/main.py
release/maixcam_copy_to_device.zip
```

## MaixCam 参数

关键参数在 [maixcam/main.py](maixcam/main.py) 顶部。

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `FRAME_W`, `FRAME_H` | `640`, `480` | 摄像头输入画面 |
| `USE_TILED_INFERENCE` | `True` | 用 320x320 分块扫描大画面，提高小目标召回 |
| `ROUND_ROBIN_TILES` | `False` | 每帧扫描全部 tile，召回优先 |
| `CONF_TH` | `0.15` | YOLO 原始候选阈值 |
| `MIN_MODEL_CONF` | `0.15` | 最低模型置信度 |
| `STRONG_MODEL_CONF` | `0.50` | 高置信目标只做纯红/橙红排除 |
| `ENABLE_COLOR_GATE` | `True` | 开启粉色/红色安全门 |
| `LOW_CONF_MIN_PINK_RATIO` | `0.03` | 低置信候选至少需要少量粉色像素 |
| `MAX_RED_BAD_RATIO` | `0.55` | 红色干扰占比阈值 |
| `REQUIRE_STABLE_FRAMES` | `2` | 连续出现多少帧才显示 |
| `DUAL_BUFF` | `False` | 检测框和当前帧对齐，适合瞄准 |

性能档位：

| 目标 | 推荐设置 | 取舍 |
| --- | --- | --- |
| 召回优先 | `ROUND_ROBIN_TILES = False` | 约 5 FPS，检测更全 |
| 平衡模式 | `ROUND_ROBIN_TILES = True`, `TILES_PER_FRAME = 3` | 帧率更高，可能漏小目标 |
| 帧率优先 | `ROUND_ROBIN_TILES = True`, `TILES_PER_FRAME = 2` | 约 10-15 FPS，召回明显下降 |

## 坐标输出

状态行：

```text
STAT,<frame>,FPS,<fps>,RAW,<raw_count>,CAND,<candidate_count>,EGGS,<target_count>,TILE,<tile_info>
```

目标行：

```text
EGG,<id>,<cx>,<cy>,<x>,<y>,<w>,<h>,<score>,<cx_norm>,<cy_norm>
```

`id` 是从左上到右下排序后的编号；`cx/cy` 是检测框中心像素坐标；`cx_norm/cy_norm` 是 0 到 1 的归一化中心点。

## 现场继续提升

最有效的提升方式是补真实数据：

1. 用最终安装角度的 MaixCam 拍摄真实场景。
2. 把漏检的卵团重新标注后加入正样本。
3. 把红色电机、线材、贴纸、手套、反光物、粉色玩具等误检对象加入负样本。
4. 重新训练、评估、转换、部署。
5. 接入执行器前，先用低功率指示灯验证坐标稳定性。

## 参考资料

- [Sipeed MaixPy](https://github.com/sipeed/maixpy)
- [MaixPy 自定义 YOLOv8 模型文档](https://wiki.sipeed.com/maixpy/doc/zh/vision/customize_model_yolov8.html)
- [MaixCam ONNX 转 MUD 文档](https://wiki.sipeed.com/maixpy/doc/zh/ai_model_converter/maixcam.html)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [Pink-Eggs Dataset V1](https://datasetninja.com/pink-eggs-dataset-v1)

