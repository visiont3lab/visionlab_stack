import numpy as np
import cv2
from cv2 import createBackgroundSubtractorKNN, createBackgroundSubtractorMOG2
#from cv2.bgsegm import createBackgroundSubtractorGMG, createBackgroundSubtractorMOG

# pip install opencv-contrib-python

# https://github.com/rajan9519/Background-subtraction
# https://github.com/tobybreckon/python-examples-cv/blob/master/mog-background-subtraction.py
# https://github.com/pranav-ust/background-subtract/blob/master/subtract.py

class backgroundSubtractionClass:
    def __init__(self,
            history = 5
        ):
        self.backSub = createBackgroundSubtractorMOG2(history=history,detectShadows=True) # 30s
        #self.backSub = createBackgroundSubtractorKNN(history=10,detectShadows=True)
        #backSub = createBackgroundSubtractorKNN(history=1000,dist2Threshold=512,detectShadows=True)
        self.foregroundMask = []
        self.backgroundEstimate = []
        
    def update(self, frame):
        self.foregroundMask = self.backSub.apply(frame)
        self.backgroundEstimate = self.backSub.getBackgroundImage()
        return self.foregroundMask,self.backgroundEstimate

if __name__ == "__main__":    
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/manuel/visiont3lab-github/public/ai_library/storage/videos/rome-1920-1080-10fps.mp4')

    cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Background Estimate",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Foreground Mask",cv2.WINDOW_NORMAL)

    bs = backgroundSubtractionClass()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    while(True):
        ret, frame = cap.read()
        if ret:
            foregroundMask, backgroundEstimate = bs.update(frame)
            #foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)

            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground Mask', foregroundMask)
            cv2.imshow('Background Estimate', backgroundEstimate)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()