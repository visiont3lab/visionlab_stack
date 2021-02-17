import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_box_custom
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
from skimage.exposure import match_histograms
from datetime import datetime

def splitImages(im):
    # 3*3 = 9  square
    num=2
    w =  int(im.shape[0]/num)
    h = int(im.shape[1]/num)

    imgwidth =  im.shape[0]
    imgheight = im.shape[1]
    
    imgs = []
    coords = []
    for j in range(0,imgheight,h):   # row
        for i in range(0,imgwidth,w): # col
            imCrop = im[i:i+w,j:j+h]
            imgs.append(imCrop)
            coords.append((i,j,w,h))
    return imgs,coords

class EstimateBackground:
    def __init__(self,img_shape):
        self.frames = []
        self.background = np.zeros(img_shape,np.uint8)
        self.buffer_size = 100
        self.sampling = 10 # we collect an image every 30 --> if 10fps --> every 3 seconds
        self.i = 0
        self.c = 0
    def updated(self, frame):
        self.i = self.i+1
        if self.i == self.sampling:
            self.i = 0
            if len(self.frames)==self.buffer_size:
                self.c = self.c +1
                self.frames.pop(0)
                self.frames.append(frame)
                if self.c == self.buffer_size:
                    self.background = np.median(self.frames, axis=0).astype(dtype=np.uint8)    
                    self.c = 0
            else:
                self.frames.append(frame)         
        return self.background

def detect(save_img=False):
    #--- MANU READ Model Background
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Remove", cv2.WINDOW_NORMAL)
    #backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelRome.png",1)
    #backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelTokyo.png",1)
    #backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelCastenaso.png",1)
    #backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelLidlImola.png",1)
    backgroundModel = cv2.imread("/home/manuel/visiont3lab-github/public/ai_library/storage/images/backgroundModelCMB.png",1)
    
    
     
    eB = EstimateBackground(backgroundModel.shape)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cv2.imshow("jk",backgroundModel)
    #cv2.waitKey(0)
    # ------

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        print("------HERE")
        print(img.shape, opt.augment)


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # --- MANU
            #cv2.imwrite("results/im.png",im0)
            detectionMask = np.zeros(backgroundModel.shape,np.uint8)
            imgMask = im0.copy()
            imgDraw = im0.copy()
            # -----

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        # --- MANU
                        imgMask,detectionMask,coord = plot_one_box_custom(xyxy, imgMask, detectionMask,label=label)
                        # ---

                        imgDraw = plot_one_box(xyxy, imgDraw, label=label, color=colors[int(cls)], line_thickness=1)


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)') 

            # Stream results
            if view_img:
                #cv2.imshow(str(p), im0 )
                #cv2.imshow("backgroundModelInit", backgroundModel)
                #backgroundModel_hsv = cv2.cvtColor(backgroundModel, cv2.COLOR_RGB2HSV)
                #im0_hsv = cv2.cvtColor(im0, cv2.COLOR_RGB2HSV)
                #cv2.imshow("b",backgroundModel_hsv[:,:,2])
                #cv2.imshow("i",im0_hsv[:,:,2])
                #a = match_histograms(backgroundModel_hsv[:,:,2],im0_hsv[:,:,2], multichannel=False)
                #cv2.imshow("br",np.uint8(a))
                #cv2.waitKey(0)
                #backgroundModel = cv2.cvtColor(backgroundModel_hsv, cv2.COLOR_HSV2RGB)  
                #cv2.imshow("backgroundModelEnd", backgroundModel)
                #cv2.waitKey(0)
                
                #detectionMaskGray = cv2.cvtColor(detectionMask,cv2.COLOR_RGB2GRAY)
                #inpaint = cv2.inpaint(imgMask,detectionMaskGray,2,cv2.INPAINT_TELEA)
                #cv2.imshow("inpainting", inpaint)
                #cv2.imshow("background", backgroundModel)
                
                #backgroundModel = match_histograms(backgroundModel,inpaint, multichannel=True)
                
                #backgroundModel_hsv = cv2.cvtColor(backgroundModel, cv2.COLOR_RGB2HSV)
                #inpaint_hsv = cv2.cvtColor(inpaint, cv2.COLOR_RGB2HSV)
                #backgroundModel_hsv[:,:,2] = match_histograms(backgroundModel_hsv[:,:,2], inpaint_hsv[:,:,2], multichannel=False)
                #backgroundModel_hsv[:,:,2] = backgroundModel_hsv[:,:,2]  - 40
                #backgroundModel = cv2.cvtColor(backgroundModel_hsv, cv2.COLOR_HSV2RGB) 
                
                '''
                imgs,coords = splitImages(inpaint)
                for c in coords:
                    i,j,w,h = c
                    backgroundModel_hsv = cv2.cvtColor(backgroundModel[i:i+w,j:j+h], cv2.COLOR_RGB2HSV)
                    im0_hsv = cv2.cvtColor(inpaint[i:i+w,j:j+h], cv2.COLOR_RGB2HSV)
                    backgroundModel_hsv[:,:,2] = match_histograms(backgroundModel_hsv[:,:,2],im0_hsv[:,:,2], multichannel=False)
                    backgroundModel[i:i+w,j:j+h] = cv2.cvtColor(backgroundModel_hsv, cv2.COLOR_HSV2RGB)  
                    #backgroundModel[i:i+w,j:j+h] = match_histograms(backgroundModel[i:i+w,j:j+h], im0[i:i+w,j:j+h], multichannel=True)
                    #cv2.imshow("img",img)
                    #cv2.waitKey(0)
                '''

                # Remove People

                #maskDetectionInv = cv2.bitwise_not(maskDetection)
                #backgroundModel = match_histograms(backgroundModel, im0, multichannel=True)
                
                #backgroundModel = eB.updated(im0)
                backgroundModelMask = cv2.bitwise_and(backgroundModel,detectionMask)
                
                now = datetime.now()
                now = now.strftime("%d_%H_%M_%S_%f")
                          
                cv2.imshow("Detection", imgDraw)
                cv2.imshow("BackgroundModel", backgroundModel)

                cv2.imshow("Image", imgMask)

                #img_hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
                #img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
                #img_hsv[:, :, 2] =  clahe.apply(img_hsv[:, :, 2])
                #res = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)    
                res = cv2.addWeighted(imgMask, 1, backgroundModelMask, 1, 0)

                cv2.imwrite("/home/manuel/visiont3lab-github/public/people-remove/images/results/Molino-Ariani-yolov5/"+now+"_yolo.png",imgDraw)
                cv2.imwrite("/home/manuel/visiont3lab-github/public/people-remove/images/results/Molino-Ariani-yolov5/"+now+"_inapaint.png",res)

                #res_hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
                #res_hsv[:, :, 2] = cv2.equalizeHist(res_hsv[:, :, 2])
                #res_hsv[:,:,0] = clahe.apply(res_hsv[:,:,0])
                #res = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2RGB)  
                #backgroundModel[i:i+w,j:j+h] = match_histograms(backgroundModel[i:i+w,j:j+h], im0[i:i+w,j:j+h], multichannel=True)
                cv2.imshow("Remove",res)
                cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
