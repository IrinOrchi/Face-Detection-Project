import numpy as np
import cv2

from sklearn import metrics
y_pred = ["a", "b", "c", "a", "b"]
y_act = ["a", "b", "c", "c", "a"]
print(metrics.confusion_matrix(y_act, y_pred, labels=["a", "b", "c"]))
print(metrics.classification_report(y_act, y_pred, labels=["a",
"b","c"]))

face_classifier = cv2.CascadeClassifier('frontalface_dataset.xml')
#eye_classifier = cv2.CascadeClassifier('/haarcascade_eye.xml')
img = cv2.imread('elon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.05, 3)
# When no faces detected, face_classifier returns and empty tuple
if faces is ():
    print("No Face Found")
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
#    eyes = eye_classifier.detectMultiScale(roi_gray)
#    for (ex, ey, ew, eh) in eyes:
#        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
 #       cv2.imshow('img', img)
#        cv2.waitKey(0)


cv2.destroyAllWindows()