from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import cv2



cap = cv2.VideoCapture('/home/manuel/visiont3lab-github/public/ai_library/storage/videos/rome-1920-1080-10fps.mp4')
backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModel.png",1)

while(True):
    ret, frame = cap.read()
    if ret:
        matched = match_histograms(backgroundModel,frame, multichannel=True)

        cv2.imshow('frame', frame)
        cv2.imshow('Matched', matched)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()