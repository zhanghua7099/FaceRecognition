import cv2
import os
import numpy as np
import sys


def generate():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite('./face/%s.pgm' % str(count), f)
            count += 1
        cv2.imshow("camera", frame)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()




def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if (sz is not None):
                        im = cv2.resize(im, (200, 200))
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError:
                    print("I/O error")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
    return [X, y]


# print(sys.argv[0])
#sys.argv获取当前目录下所有文件名称，并将其存储在数组中

def face_rec():
    names = ['Zhang', 'Jane', 'Jack']
    X, y = [], []
    im1 = cv2.imread('./face/1.pgm', cv2.IMREAD_GRAYSCALE)
    im1 = cv2.resize(im1, (200, 200))
    X.append(np.asarray(im1, dtype=np.uint8))
    y.append(0)
    im2 = cv2.imread('./face/2.pgm', cv2.IMREAD_GRAYSCALE)
    im2 = cv2.resize(im2, (200, 200))
    X.append(np.asarray(im2, dtype=np.uint8))
    y.append(0)
    im3 = cv2.imread('./face/3.pgm', cv2.IMREAD_GRAYSCALE)
    im3 = cv2.resize(im3, (200, 200))
    X.append(np.asarray(im3, dtype=np.uint8))
    y.append(0)
    im4 = cv2.imread('./face/4.pgm', cv2.IMREAD_GRAYSCALE)
    im4 = cv2.resize(im4, (200, 200))
    X.append(np.asarray(im4, dtype=np.uint8))
    y.append(0)
    im5 = cv2.imread('./face/5.pgm', cv2.IMREAD_GRAYSCALE)
    im5 = cv2.resize(im5, (200, 200))
    X.append(np.asarray(im5, dtype=np.uint8))
    y.append(0)
    im6 = cv2.imread('./face/6.pgm', cv2.IMREAD_GRAYSCALE)
    im6 = cv2.resize(im6, (200, 200))
    X.append(np.asarray(im6, dtype=np.uint8))
    y.append(0)
    im7 = cv2.imread('./face/7.pgm', cv2.IMREAD_GRAYSCALE)
    im7 = cv2.resize(im7, (200, 200))
    X.append(np.asarray(im7, dtype=np.uint8))
    y.append(0)
    y = np.asarray(y, dtype=np.int32)
    #model = cv2.face.createEigenFaceRecognizer()
    #model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while(True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200,200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                # cv2.putText(img, names[0], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow("camera", img)
        if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


face_rec()
