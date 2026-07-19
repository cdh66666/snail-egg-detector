from maix import app, camera, display, image, nn, time


MODEL = "/root/models/yolo11n.mud"
detector = nn.YOLO11(model=MODEL, dual_buff=False)
FRAME_W = detector.input_width()
FRAME_H = detector.input_height()


print("SNAIL EGG CAMERA PREVIEW BOOT")
cam = camera.Camera(FRAME_W, FRAME_H)
disp = display.Display()
frame_id = 0

while not app.need_exit():
    img = cam.read()
    objects = detector.detect(img, conf_th=0.2, iou_th=0.45)
    for obj in objects:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=image.COLOR_GREEN, thickness=2)
    fps = time.fps()
    if frame_id % 30 == 0:
        print("MINIMAL,FRAME,%d,FPS,%.1f,OBJECTS,%d" % (frame_id, fps, len(objects)), flush=True)
    img.draw_string(2, 2, "PREVIEW %.1f FPS" % fps, image.COLOR_GREEN)
    disp.show(img)
    frame_id += 1

print("SNAIL EGG CAMERA PREVIEW STOP")
