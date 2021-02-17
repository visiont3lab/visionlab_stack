import os
from PIL import Image
import numpy as np
import time
import cv2 
import time
from datetime import datetime

from visionlab_stack.detection.yolov5.yolov5Lib import yolov5Class
from visionlab_stack.inpainting.inpainting.inpaintingLib import inpaintingClass
from visionlab_stack.vision.backgroundSubtraction.backgroundSubtractionLib import backgroundSubtractionClass
from visionlab_stack.utils.common import splitImage,increaseRoi,createImageHole,createMask

class peopleRemove:

    def __init__(self):

        # Cpu and gpu
        self.detect = yolov5Class( param_device='cpu', # '0'
                        param_img_size = 1920,
                        param_confidence_th = 0.15,
                        param_iou_th = 0.3,
                        classes = [0],
                        weights = "./data/models/detection/yolov5/yolov5s.pt"
                    )
      
        # Cpu and Gpu
        self.inpaint = inpaintingClass(
                        inpaint_yml = "./data/models/inpainting/inpainting/inpaint.yml",
                        inpaint_checkpoint = "./data/models/inpainting/inpainting/release_places2_256_deepfill_v2",
                        inpaint_size = [256,256]
                    )
        
        # Back Subtractor
        self.backSub = backgroundSubtractionClass(history=5)

    def run_detection_yolov5(self, im):

        imgMask= im.copy()
        imgDraw=im.copy()
        mask = np.zeros((im.shape[0],im.shape[1],3), np.uint8)

        coords,scores =self.detect.doDetection(im)

        if(len(coords)):
            for coord,score in zip(coords,scores):
                y,x,h,w = coord
                
                # Enlarge mask
                #halfw = int(w/2)
                #halfh = int(h/2)
                #axes = (x+halfw,y+halfh)
                #mask = cv2.ellipse(mask,axes , (halfw,halfh), 0, 0, 360, (255,255,255), -1)
                coordMask = increaseRoi(coord,0.5)                 
                my,mx,mh,mw = coordMask
                mask = cv2.rectangle(mask, (mx,my),(mx+mw,my+mh), (255,255,255), -1)
              
                # Drawings
                cv2.rectangle(imgDraw, (x, y), (x+w, y+h), [0,255,0], 1)
                text = "{}: {:.4f}".format("Person",score)
                cv2.putText(imgDraw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0], 1)
                imgMask = cv2.rectangle(imgMask, (x,y),(x+w,y+h), (0,0,0), -1)

                #createImageHole(img_hole,coord[1], coord[0], coord[1] + coord[3], coord[0] + coord[2])
                #createMask(mask,coord[1], coord[0], coord[1] + coord[3], coord[0] + coord[2])
        
        return imgDraw,imgMask,mask

    def run_inpainting(self,im,mask):
        #cv2.imshow("im",im)
        #cv2.imshow("mask",mask)
        #cv2.waitKey(33) 
        #INPAINTING
        #apply inpainting on single image splitted
        #selezionare dimensione immagine da passare alla rete di inpainting        
        h = im.shape[0]
        w = im.shape[1]
        res = im.copy() #np.zeros(im.shape, dtype=np.uint8) #Image.new('RGB', (w,h))      
        heightInpainting=256 #self.heightYolo
        widthInpainting=256 #int(h/4) #self.widthYolo
        listCoordinatesSplit=splitImage(h,w,heightInpainting,widthInpainting,overlap=10)
        for index,coordinates in enumerate(listCoordinatesSplit):
            crop_img=im[coordinates[0]:coordinates[0]+heightInpainting,coordinates[1]:coordinates[1]+widthInpainting]         
            crop_mask=mask[coordinates[0]:coordinates[0]+heightInpainting,coordinates[1]:coordinates[1]+widthInpainting] 
            if 255 in crop_mask:
                img_Inpaint = self.inpaint.doInpainting(crop_img, crop_mask)
                #img_Inpaint = cv2.inpaint(crop_img,cv2.cvtColor(crop_mask,cv2.COLOR_RGB2GRAY),5,cv2.INPAINT_TELEA)
                res[coordinates[0]:coordinates[0]+heightInpainting,coordinates[1]:coordinates[1]+widthInpainting]= img_Inpaint       
        return res

    def run_onfolder(self, input_folder, output_folder):
        names = os.listdir(input_folder)
        names.sort() 
        for name in names:
            filename_inp = os.path.join(input_folder,name)
            name_without_ext = os. path. splitext(name)[0]

           
            startTotal = time.time()
            image=cv2.imread(filename_inp,cv2.IMREAD_COLOR) 


            # --- DETECTION
            imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov5(image)

            # ---- INPAINTING
            imgDrawInpaint = self.run_inpainting(image,maskDet )
            
            
            stopTotal = time.time()
            time_to_process = str(round(stopTotal-startTotal,3))
            print("Total time image "+filename_inp+"= "+time_to_process+"s")

            cv2.imwrite(output_folder+"/images/"+name_without_ext+".png",image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
            cv2.imwrite(output_folder+"/masks/"+name_without_ext+".png",maskDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
            cv2.imwrite(output_folder+"/detection/"+name_without_ext+".png",imgDrawDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
            cv2.imwrite(output_folder+"/inpainting/"+name_without_ext+".png",imgDrawInpaint, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
  
    def run_onvideo(self, path_video, output_folder):
        cap = cv2.VideoCapture(path_video)
        while(True):
            now = datetime.now()
            s = now.strftime("%Y_%d_%H-%M-%S-%f")
            ret, image = cap.read()
            if ret:
                startTotal = time.time()

                # ---- DETECTIONÇ
                imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov5(image)

                # ----- INPAINTING
                imgDrawInpaint = self.run_inpainting(image,maskDet )


                stopTotal = time.time()
                time_to_process = str(round(stopTotal-startTotal,3))
                print("Total time image "+s+"= "+time_to_process+"s")
                
                foregroundMask,backgroundEstimate = self.backSub.update(imgDrawInpaint)
                #masksArea = cv2.bitwise_and(imgDrawInpaint,maskDet)
                #maskDet = cv2.dilate(maskDet, kernel=np.ones((9,9), np.uint8) , iterations=1) 
                #backgroundArea = cv2.bitwise_and(imgDrawInpaint,cv2.bitwise_not(maskDet))
                #masksArea = cv2.bitwise_and(backgroundEstimate,maskDet)
                #estimate = cv2.bitwise_or(backgroundArea,masksArea)
                #estimate =cv2.addWeighted(backgroundArea,0.5,masksArea,0.5,0)
                #cv2.imshow("Mask",masksArea)
                #cv2.imshow("sbe",backgroundArea)
           
                cv2.imwrite(output_folder+"/images/"+s+".png",image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                cv2.imwrite(output_folder+"/masks/"+s+".png",maskDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                cv2.imwrite(output_folder+"/detection/"+s+".png",imgDrawDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                cv2.imwrite(output_folder+"/inpainting/"+s+".png",imgDrawInpaint, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                cv2.imwrite(output_folder+"/bg_estimate/"+s+".png",backgroundEstimate, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                
                #cv2.imwrite(output_folder+"/images/"+s+".png",image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                #cv2.imwrite(output_folder+"/masks/"+s+".png",maskDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    input_folder =  os.path.join(".","data","images")
    output_folder =  os.path.join(".","data","results")
    video_path = os.path.join(".", "data", "videos","rome-1920-1080-10fps-short.mp4" )
    pr = peopleRemove()
    
    
    pr.run_onfolder(input_folder=input_folder, output_folder=output_folder)
    # pr.run_onvideo(path_video =video_path,output_folder=output_folder)

