import cv2
import sys
import numpy as np


def showWindow(image):
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


face_cascade = cv2.CascadeClassifier("D:/python/openCV/openCV/openCV/haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("D:/python/example/full/0a8ea8d08ae6c0a48fece080daae4ebd3789e6c0.jpg")
eye_cascade = cv2.CascadeClassifier("D:/python/openCV/openCV/openCV/haarcascade_eye.xml")
# img_path = sys.argv[1]
# print("------", img_path)
img = cv2.imread("D:/python/openCV/openCV/openCV/aa.jpg")
# img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow("Image")
# cv2.imshow("Image", img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# High Pass
# blur = cv2.blur(img, (5, 5))
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()

# gray_lap = cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3)
# dst = cv2.convertScaleAbs(gray_lap)
# cv2.imshow('laplacian', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
showWindow(img_gray)
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
for (x, y, w, h) in faces:
    print("-------", "face")
    face_area = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face_gray = img_gray[y:y + h, x:x + w]
    face_img = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(face_gray)
    for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # eh = eh / 2
        cv2.circle(face_img, center=(int(ex + ew / 2), int(ey + eh / 2)), radius=int(eh / 2), color=(0, 255, 0))
showWindow(img)
cv2.imwrite("E:/迅雷下载/aaa.png", img)
