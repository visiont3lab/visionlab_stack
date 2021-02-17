import numpy as np
import cv2
from cv2 import createBackgroundSubtractorKNN, createBackgroundSubtractorMOG2

# https://github.com/rajan9519/Background-subtraction
# https://github.com/tobybreckon/python-examples-cv/blob/master/mog-background-subtraction.py
# https://github.com/pranav-ust/background-subtract/blob/master/subtract.py

backSub = createBackgroundSubtractorMOG2(history=300) # 30s
#backSub = createBackgroundSubtractorKNN()
#backSub = createBackgroundSubtractorKNN(history=1000,dist2Threshold=512,detectShadows=True)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/manuel/visiont3lab-github/public/ai_library/storage/videos/rome-1920-1080-10fps.mp4')

cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
cv2.namedWindow("FG Mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("Background Mask",cv2.WINDOW_NORMAL)
cv2.namedWindow("Background Median",cv2.WINDOW_NORMAL)

i=0
frames = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    foregroundMask = backSub.apply(frame)
    background = backSub.getBackgroundImage()

    if i==10:
        frames.append(frame)
        medianBackground = np.median(frames, axis=0).astype(dtype=np.uint8)    
        cv2.imshow('Background Median', medianBackground)
        if len(frames) == 30:
            frames = []
        i=0
    i=i+1

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', foregroundMask)
    cv2.imshow('Background Mask', background)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()