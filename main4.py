import cv2
import numpy as np

from sklearn import metrics
y_pred = ["a", "b", "c", "a", "b"]
y_act = ["a", "b", "c", "c", "a"]
print(metrics.confusion_matrix(y_act, y_pred, labels=["a", "b", "c"]))
print(metrics.classification_report(y_act, y_pred, labels=["a",
"b","c"]))

img=cv2.imread('elon2.jpg', cv2.IMREAD_UNCHANGED)
print('original dimensions: ', img.shape)

scale_percent=80
width=int(img.shape[1]*scale_percent/100)
height=int(img.shape[0]*scale_percent/100)
dim = (width, height)
resized=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimension: ', resized.shape)
cv2.imshow("Resized Image: ", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


face_classifier = cv2.CascadeClassifier('frontalface_dataset.xml')
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
'''When no faces detected, face_classifier returns and empty tuple'''
if faces is ():
    print("No faces found")
for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x + w, y + h), (127, 0, 255), 2)
    cv2.imshow('Face Detection', resized)
    cv2.waitKey(0)



cv2.destroyAllWindows()