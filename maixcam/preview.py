from maix import app, camera, display, image, time


FRAME_W = 320
FRAME_H = 240


print("SNAIL EGG CAMERA PREVIEW BOOT")
cam = camera.Camera(FRAME_W, FRAME_H)
disp = display.Display()

while not app.need_exit():
    img = cam.read()
    img.draw_string(2, 2, "PREVIEW %.1f FPS" % time.fps(), image.COLOR_GREEN)
    disp.show(img)

print("SNAIL EGG CAMERA PREVIEW STOP")
