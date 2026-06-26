import cv2
from effects import blur 
from detector import HandDetector
import pyvirtualcam

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector()

virtual_cam = pyvirtualcam.Camera(
    width=1280,
    height=720,
    fps=30,
    backend="obs"
)

print("Backend:", virtual_cam.backend)

print("Virtual Camera Ready!")

while True:

    ret, frame = cam.read()

    if not ret:
        break

    frame, peace = detector.detect(frame)

    if peace:
        frame = blur(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    virtual_cam.send(rgb)
    virtual_cam.sleep_until_next_frame()

    cv2.imshow("Foto Kita Blur", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()