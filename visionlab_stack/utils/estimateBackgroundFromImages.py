
# https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/

import numpy as np
import cv2
import os

path = "/home/manuel/visiont3lab-github/public/people-remove/images/input/Molino-Ariani"
paths = os.listdir(path)
chosen = np.array( len(paths)* np.random.uniform(size=int(len(paths)/2)) , dtype=np.int32)
i=0
frames = []
for p in paths:
    if i in chosen:
        fp = os.path.join(path,p)
        frame = cv2.imread(fp,1)
        frames.append(frame)
    i = i+1

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.imwrite("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelMolinoAriani.png", medianFrame)
cv2.waitKey(0)



