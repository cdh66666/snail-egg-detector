# VSCode + SSH MaixCam 工作流

MaixCam 能通过 SSH 访问后，可以不用 MaixVision 手工复制文件，直接从 VSCode 或 PowerShell 自动部署。

## 1. 获取 MaixCam 地址

在 MaixCam 上打开系统信息，记录：

- `maixcam-xxxx.local`
- 或 IP 地址，例如 `192.168.10.107`

如果 Windows 不能解析 `.local`，直接使用 IP。

## 2. 设置连接信息

在仓库根目录运行：

```powershell
$env:MAIXCAM_HOST='192.168.10.107'
$env:MAIXCAM_PASSWORD='root'
```

默认账号通常是 `root/root`。如果你改过密码，把 `MAIXCAM_PASSWORD` 换成新密码。

## 3. 探测设备

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 probe
```

成功时会显示主机名、Python 版本和模型文件。

## 4. 调试运行

第一次上传主程序和模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 deploy-run
```

模型已经在板子上时，只上传主程序：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 deploy-run --skip-models
```

只跑 20 秒并把日志打到终端：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 deploy-run --skip-models --run-seconds 20
```

## 5. 安装为开机自启

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 install-autostart --app-id cdh1_ --reboot
```

这会写入：

```text
/maixapp/apps/cdh1_/main.py
/maixapp/auto_start.txt
/root/models/snail_eggs_yolov8n_320x320.mud
/root/models/snail_eggs_yolov8n_320x320.cvimodel
```

## 6. VSCode 任务

打开仓库目录后，使用 `Terminal -> Run Task`：

- `MaixCam: Probe`
- `MaixCam: Deploy main + models + run`
- `MaixCam: Deploy main only + run`
- `MaixCam: Deploy only`
- `MaixCam: Run 20s for Codex debug`
- `MaixCam: Reboot board`

## 7. 常见问题

如果板子报：

```text
mmf add vi channel failed
No buffer space available
CVI_SYS_Bind(VI-VPSS) failed
```

通常是摄像头资源被上一次进程占住。先重启板子：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\maix_remote.ps1 reboot
```

如果 SSH 端口能连上但一直不返回 banner，等 30 到 120 秒再试。MaixCam 刚开机时网络服务可能比端口开放更晚完全可用。

调参时保持激光断开，先用 `RUN_MODE = 0/1/3` 看原始候选和颜色过滤效果，稳定后再切回 `RUN_MODE = 2`。
