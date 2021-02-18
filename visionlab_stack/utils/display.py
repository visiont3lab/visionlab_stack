import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import cv2
import numpy as np

def images():
    folder = "/home/manuel/visiont3lab-github/public/ai_library/visionlab_stack/data/results-molino-ariani"
    path_images = sorted(os.listdir(os.path.join(folder, "images")))
    sample_image = cv2.imread(os.path.join(folder,"images",path_images[0]),1)
    h,w = sample_image.shape[0], sample_image.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/manuel/visiont3lab-github/public/ai_library/visionlab_stack/data/videos/results-molino-ariani.mp4', fourcc, 1.0, (w*2,h))

    for p in path_images:
        #image = cv2.imread(os.path.join(folder,"images",p),1)
        #mask = cv2.imread(os.path.join(folder,"masks",p),1)
        inpainting = cv2.imread(os.path.join(folder,"inpainting",p),1)
        detection = cv2.imread(os.path.join(folder,"detection",p),1)
        
        #cv2.putText(image, 'Image', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #cv2.putText(mask, 'Mask', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(inpainting, 'Inpainting', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(detection, 'Detection', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        res = np.concatenate((detection, inpainting), axis=1) # vertically
        #cv2.imwrite(os.path.join(folder,p),res)
        
        #res = cv2.resize(res, (w,h))
        #cv2.imshow("im", res)
        #cv2.waitKey(0)

        out.write(res)

    out.release()

def video():
    folder = "/home/manuel/visiont3lab-github/public/ai_library/visionlab_stack/data/results"
    path_images = sorted(os.listdir(os.path.join(folder, "images")))
    sample_image = cv2.imread(os.path.join(folder,"images",path_images[0]),1)
    h,w = sample_image.shape[0], sample_image.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/manuel/visiont3lab-github/public/ai_library/visionlab_stack/data/videos/rome-results.mp4', fourcc, 10.0, (w,h))

    for p in path_images:
        image = cv2.imread(os.path.join(folder,"images",p),1)
        mask = cv2.imread(os.path.join(folder,"masks",p),1)
        inpainting = cv2.imread(os.path.join(folder,"inpainting",p),1)
        detection = cv2.imread(os.path.join(folder,"detection",p),1)
        bg_estimate = cv2.imread(os.path.join(folder,"bg_estimate",p),1)

        cv2.putText(image, 'Image', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(mask, 'Mask', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(inpainting, 'Inpainting', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(detection, 'Detection', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(bg_estimate, 'Background Estimate', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        im1 = np.concatenate((image, detection), axis=1) # horizontally
        im2 = np.concatenate((inpainting, bg_estimate), axis=1) # horizontally
        res = np.concatenate((im1, im2), axis=0) # vertically

        res = cv2.resize(res, (w,h))
        #cv2.imshow("im", res)
        #cv2.waitKey(0)

        out.write(res)

    out.release()

images()