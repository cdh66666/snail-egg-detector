# MaixCAM 实机回归记录：2026-07-20

设备：`maixcam-b226`，`192.168.10.107`，MaixPy `4.12.5`。

## 实测版本

- 模型：`snail_eggs_yolo11n_320x224_v26.cvimodel`
- 输入：`320x224`，相机采集请求 `60 FPS`
- `camera.Camera(..., buff_num=1)`
- YOLO11 `dual_buff=True`
- 每 2 个相机帧执行一次检测，中间帧由轨迹器维护
- 颜色复核：YOLO 后置的轻量粉色/红色 sanity check，最多 2/4 个采样点
- 网页流：独立 JPEG streamer，端口 `8001`
- 激光：测试期间通过急停关闭，未执行高功率消杀

## 三轮实机数据

| 版本 | loop FPS | detect Hz | detect stage | stream FPS | eggs |
| --- | ---: | ---: | ---: | ---: | ---: |
| v26，双缓冲，每帧检测 | 3.6--4.0 | 3.8--4.1 | 109--116 ms | 3.8--4.1 | 8--9 |
| 性能补丁，单缓冲、隔帧检测 | 5.8--6.5 | 2.7--3.4 | 113--117 ms | 5.3--7.0 | 8--9 |
| 当前候选，双缓冲、隔帧、轻量颜色复核 | 5.6--6.3 | 2.8--3.5 | 96--111 ms | 5.6--7.0 | 8--9 |

## 结论

当前瓶颈不是相机采集，而是 MaixPy Python 推理调用及其后置处理路径。官方文档的 YOLO11 基准是平台/模型基准，不能直接当成本项目端到端帧率；本项目还包含检测后过滤、跟踪、网页 JPEG 编码和安全控制。当前版本优先保留识别结果和轨迹连续性，不能宣称已经达到 30/60 FPS。

下一轮优化必须使用同一设备上的 `PROF` 数据做 A/B：分别测模型调用、颜色复核、网页编码和跟踪控制，任何速度提升都要同时检查 `eggs`、漏检帧和锁定身份是否退化。

参考：

- Sipeed MaixPy 双缓冲说明：<https://wiki.sipeed.com/maixpy/doc/en/vision/dual_buff.html>
- Sipeed MaixPy 相机低延迟与帧率说明：<https://wiki.sipeed.com/maixpy/doc/en/vision/camera.html>
- Pomacea canaliculata egg-mass image-processing study：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11599704/>
