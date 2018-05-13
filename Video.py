import cv2

camera = cv2.VideoCapture(0)
count = 0
while (True):
    ret, frame = camera.read()
    cv2.imshow("camera", frame)
    if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()