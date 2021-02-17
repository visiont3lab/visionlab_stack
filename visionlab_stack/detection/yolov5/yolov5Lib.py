import sys
sys.path.append('./visionlab_stack/detection/yolov5') 

import numpy as np
import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, \
    non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_box_custom
from utils.torch_utils import select_device, load_classifier, time_synchronized
#from skimage.exposure import match_histograms
from datetime import datetime

class yolov5Class:
    
    def __init__(self,
                param_device = 'cpu',
                param_img_size = 1920,
                param_confidence_th = 0.15,
                param_iou_th = 0.3,
                classes = [0],
                weights = "./data/models/detection/yolov5/yolov5s.pt"
        ):
        self.param_device = param_device #'0'
        self.param_img_size = param_img_size
        self.param_confidence_th = param_confidence_th
        self.param_iou_th = param_iou_th
        self.classes = classes # person
        #self.weights = "visionlab_stack/detection/yolov5/weights/yolov5m.pt"
        self.weights = weights  #"visionlab_stack/detection/yolov5/weights/yolov5s.pt"
        
        self.augment = False
        self.agnostic = False

        # Initialize
        set_logging()
        self.device = select_device(device=self.param_device) # '0' GPU
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        #self.model = load_model(self.weights, map_location=self.device)  # load FP32 model
        
        self.param_stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size= self.param_img_size, s=self.param_stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.t0 = time.time()

    def doDetection(self, image):
        # Preprocess
        #image = np.moveaxis(image, -1, 0)
        im = letterbox(image, self.imgsz, stride=self.param_stride)[0]

        img = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
      
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        #pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = non_max_suppression(pred, self.param_confidence_th, self.param_iou_th, classes=self.classes, agnostic=self.agnostic)
        t2 = time_synchronized()
        # Process detections
        im0 = image.copy() 
        coords = []
        scores = []

        detectionMask = np.zeros(im0.shape,np.uint8)
        imgMask = im0.copy()
        imgDraw = im0.copy()
        
        for i, det in enumerate(pred):  # detections per image
            # --- MANU
            #cv2.imwrite("results/im.png",im0)
            # -----
            now = datetime.now()
            s = now.strftime("%d_%H_%M_%S_%f -- ")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f" {n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # --- MANU
                    imgMask,detectionMask,coord = plot_one_box_custom(xyxy, imgMask, detectionMask,label=label)
                    coords.append(coord)
                    scores.append(conf)
                    # ---
                    imgDraw = plot_one_box(xyxy, imgDraw, label=label, color=self.colors[int(cls)], line_thickness=1)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)') 
            
            # Plot
            #backgroundModelMask = cv2.bitwise_and(backgroundModel,detectionMask)
            #res = cv2.addWeighted(imgMask, 1, backgroundModelMask, 1, 0)

            #cv2.imshow("Detection", imgDraw)
            #cv2.imshow("BackgroundModel", backgroundModel)
            #cv2.imshow("Image", detectionMask)
            #cv2.imwrite("/home/manuel/visiont3lab-github/public/people-remove/images/results/Molino-Ariani-yolov5/"+now+"_yolo.png",imgDraw)
            #cv2.imwrite("/home/manuel/visiont3lab-github/public/people-remove/images/results/Molino-Ariani-yolov5/"+now+"_inapaint.png",res)
            #cv2.imshow("Remove",res)
            #cv2.waitKey(0) 
        return coords,scores
