import cv2
from matplotlib import pyplot as plt
import os
import uuid


capture = cv2.VideoCapture(0)
while capture.isOpened(): 
    ret, frame = capture.read()

    frame = frame[150:150+250,250:250+250, :]

    cv2.imshow('image collection', frame)
    
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(os.path.join('data', 'pos'), '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
        
    if cv2.waitKey(1) & 0XFF == ord('a'):
        imgname = os.path.join(os.path.join('data', 'anc'), '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

capture.release
cv2.destoryAllWindows()