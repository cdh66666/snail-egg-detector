# MaixCam SSH 远程控制极简教程

这份教程用于远程控制 MaixCam 视觉摄像头：登录设备、上传福寿螺卵识别程序、启动/停止程序、查看日志、设置开机自启。

## 1. 先确认 MaixCam 的 IP

MaixCam 和电脑要在同一个网络里。假设 MaixCam IP 是：

```text
192.168.10.107
```

Windows PowerShell / Mac 终端都可以先测试：

```bash
ping 192.168.10.107
```

能 ping 通，再 SSH 登录：

```bash
ssh root@192.168.10.107
```

默认密码通常是：

```text
root
```

登录成功后，你看到的是 MaixCam 里面的 Linux 命令行。

## 2. 最常用的几个命令

查看当前在哪：

```bash
pwd
```

查看当前目录文件：

```bash
ls
```

进入福寿螺识别 app 目录：

```bash
cd /maixapp/apps/snail_egg
```

查看识别程序是否存在：

```bash
ls -lh /maixapp/apps/snail_egg/main.py
```

查看模型文件：

```bash
ls -lh /root/models/snail_eggs_yolov8n_640x480.*
```

## 3. 从电脑上传程序到 MaixCam

在电脑终端里执行，不是在 SSH 登录后的 MaixCam 里执行。

Windows PowerShell：

```powershell
$MAIX_IP = "192.168.10.107"
$COURSE = "C:\Users\admin\Documents\GitHub\snail-egg-detector"

ssh root@${MAIX_IP} "mkdir -p /maixapp/apps/snail_egg /root/models"

scp "$COURSE\maixcam\main.py" `
  "root@${MAIX_IP}:/maixapp/apps/snail_egg/main.py"
```

Mac / Linux：

```bash
MAIX_IP=192.168.10.107

ssh root@$MAIX_IP "mkdir -p /maixapp/apps/snail_egg /root/models"
scp maixcam/main.py root@$MAIX_IP:/maixapp/apps/snail_egg/main.py
```

如果还要上传模型：

```bash
scp release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.* \
  root@$MAIX_IP:/root/models/
```

## 4. 远程启动福寿螺识别程序

登录 MaixCam 后手动运行：

```bash
cd /maixapp/apps/snail_egg
python3 main.py
```

这种方式适合调试，因为报错会直接显示在终端里。

如果想让它在后台运行：

```bash
cd /maixapp/apps/snail_egg
nohup python3 main.py auto_start > /tmp/cdh1.log 2>&1 &
```

查看日志：

```bash
tail -f /tmp/cdh1.log
```

按 `Ctrl + C` 只是退出看日志，不会停止后台程序。

## 5. 停止正在运行的识别程序

查看进程：

```bash
ps | grep main.py
```

停止福寿螺识别程序：

```bash
pkill -f /maixapp/apps/snail_egg/main.py
```

如果没有 `pkill`，就先查 PID：

```bash
ps | grep main.py
```

然后杀掉对应 PID：

```bash
kill 进程号
```

## 6. 设置开机自启

MaixCam 的自启动文件是：

```text
/maixapp/auto_start.txt
```

让福寿螺识别程序开机自动启动：

```bash
echo snail_egg > /maixapp/auto_start.txt
sync
reboot
```

这里的 `snail_egg` 是 app ID，对应这个目录：

```text
/maixapp/apps/snail_egg/main.py
```

重启后检查：

```bash
cat /maixapp/auto_start.txt
ps | grep main.py
```

看到 `snail_egg` 和 `python3 /maixapp/apps/snail_egg/main.py auto_start`，说明自启动成功。

## 7. 取消开机自启

如果不想上电自动运行：

```bash
echo "" > /maixapp/auto_start.txt
sync
reboot
```

## 8. 远程重启 MaixCam

```bash
reboot
```

或者从电脑直接执行：

```bash
ssh root@192.168.10.107 "reboot"
```

## 9. 远程做一次摄像头/程序检查

查看是否有摄像头画面相关进程：

```bash
ps | grep python3
```

查看最近日志：

```bash
tail -80 /tmp/cdh1.log
```

查看模型路径是否正确：

```bash
grep "MODEL =" /maixapp/apps/snail_egg/main.py
```

当前程序应该指向：

```text
/root/models/snail_eggs_yolov8n_640x480.mud
```

## 10. 常见问题

### SSH 连不上

先检查：

```bash
ping 192.168.10.107
```

如果 ping 不通：

- 确认 MaixCam 和电脑在同一 Wi-Fi 或同一有线网络。
- 确认 IP 没变。
- 用 MaixCam 屏幕或 MaixVision 查看当前 IP。

### 密码不对

默认通常是：

```text
root
```

如果改过密码，就用你改过的密码。

### 上传后没效果

确认你上传的是这个路径：

```text
/maixapp/apps/snail_egg/main.py
```

确认当前正在运行的也是这个文件：

```bash
ps | grep main.py
```

### 程序一运行就退出

前台运行看报错：

```bash
cd /maixapp/apps/snail_egg
python3 main.py
```

常见原因：

- 模型文件没放到 `/root/models/`。
- `.mud` 和 `.cvimodel` 文件名不匹配。
- 摄像头被其他程序占用。
- 代码里使用了当前固件不支持的 API。

## 11. 一套最短部署命令

Windows PowerShell：

```powershell
$MAIX_IP = "192.168.10.107"
$ROOT = "C:\Users\admin\Documents\GitHub\snail-egg-detector"

ssh root@${MAIX_IP} "mkdir -p /maixapp/apps/snail_egg /root/models"
scp "$ROOT\maixcam\main.py" "root@${MAIX_IP}:/maixapp/apps/snail_egg/main.py"
scp "$ROOT\release\maixcam_copy_to_device\root\models\snail_eggs_yolov8n_640x480.*" "root@${MAIX_IP}:/root/models/"
ssh root@${MAIX_IP} "echo snail_egg > /maixapp/auto_start.txt && sync && reboot"
```

Mac / Linux：

```bash
MAIX_IP=192.168.10.107

ssh root@$MAIX_IP "mkdir -p /maixapp/apps/snail_egg /root/models"
scp maixcam/main.py root@$MAIX_IP:/maixapp/apps/snail_egg/main.py
scp release/maixcam_copy_to_device/root/models/snail_eggs_yolov8n_640x480.* root@$MAIX_IP:/root/models/
ssh root@$MAIX_IP "echo snail_egg > /maixapp/auto_start.txt && sync && reboot"
```
