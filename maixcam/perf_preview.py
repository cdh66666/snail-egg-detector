from maix import app, camera, display, time

CAM_W = 320
CAM_H = 256
cam = camera.Camera(CAM_W, CAM_H, buff_num=3)
disp = display.Display()
frame_id = 0
print("PERF_PREVIEW_START,%dx%d" % (CAM_W, CAM_H))
while not app.need_exit() and frame_id < 360:
    img = cam.read()
    disp.show(img)
    if frame_id % 60 == 0:
        print("PERF_PREVIEW,FRAME,%d,FPS,%.1f" % (frame_id, time.fps()))
    frame_id += 1
print("PERF_PREVIEW_DONE,FRAME,%d,FPS,%.1f" % (frame_id, time.fps()))
