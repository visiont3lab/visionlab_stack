
# https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/

import numpy as np
import cv2
from skimage import data, filters

# pip install scikit-image

# Open Video
cap = cv2.VideoCapture('/home/manuel/visiont3lab-github/public/ai_library/storage/videos/rome-1920-1080-10fps.mp4')
#cap = cv2.VideoCapture('/home/manuel/visiont3lab-github/public/ai_library/storage/videos/tokyo.mp4')


# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)

#frameIds = 3200 * np.random.uniform(size=100)


# Store selected frames in an array
frames = []
for fid in frameIds:
#for fid in range(0,50*200,200):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.imwrite("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelRome.png", medianFrame)
cv2.waitKey(0)


# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True
while(ret):

    # Read frame
    ret, frame = cap.read()
    # Convert current frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and 
    # the median frame
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    # Display image
    cv2.imshow('frame', dframe)
    cv2.waitKey(20)

# Release video object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()


