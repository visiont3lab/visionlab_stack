import os
from PIL import Image
import numpy as np
import time
import cv2 
import time
from datetime import datetime

from detection.yolov3.yolov3Lib import yolov3Class
from detection.yolov5.yolov5Lib import yolov5Class
#from detection.detectron2.detectron2Lib import detectron2Class
from inpainting.inpainting.inpaintingLib import inpaintingClass
from utils.common import splitImage,increaseRoi,createImageHole,createMask

class peopleRemove:

    def __init__(self):
        #self.detect=detectron2Class()
        #self.detect=yolov3Class()   
        self.detect=yolov5Class()  
        self.Inpaint = inpaintingClass() # split images 
    
    def run_detection_yolov3(self, im):

        heightYolo = 608
        widthYolo = heightYolo
        overlapImage = 50

        imgMask= im.copy()
        imgDraw=im.copy()
        mask = np.zeros((im.shape[0],im.shape[1],3), np.uint8)

        allCoords=[]
        allScores=[]

        #First yolo detection on whole image. Maybe we can use a different threshold detection
        coords,scores =self.detect.doDetection(im,[0,0],size=(heightYolo,widthYolo),confidence_th=0.6)
        if(coords):
            allCoords.extend(coords)
            allScores.extend(scores)
            
        # Split image approach
        listImages=splitImage(im.shape[0],im.shape[1],heightYolo,widthYolo,overlapImage)
        for index,imCoords in enumerate(listImages):
            crop_img = im[imCoords[0]:imCoords[0]+heightYolo,imCoords[1]:imCoords[1]+widthYolo]      
            coords,scores =self.detect.doDetection(crop_img,imCoords,size=(heightYolo,widthYolo))
            if(len(coords)):  
                allCoords.extend(coords)
                allScores.extend(scores)
        
        # Collect coordinates
        if(len(allCoords)):
            for coord,score in zip(allCoords,allScores):
                y,x,h,w = coord
                
                # Enlarge mask
                coordMask = increaseRoi(coord,0.5)                 
                my,mx,mh,mw = coordMask
                mask = cv2.rectangle(mask, (mx,my),(mx+mw,my+mh), (255,255,255), -1)
              
                # Drawings
                cv2.rectangle(imgDraw, (x, y), (x+w, y+h), [0,255,0], 1)
                text = "{}: {:.4f}".format("Person",score)
                cv2.putText(imgDraw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,255,0], 1)
                imgMask = cv2.rectangle(imgMask, (x,y),(x+w,y+h), (0,0,0), -1)
                

        return imgDraw,imgMask,mask

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
        newImageInpainting = Image.new('RGB', (w,h))      
        heightInpainting=608 #self.heightYolo
        widthInpainting=608 #int(h/4) #self.widthYolo
        listCoordinatesSplit=splitImage(h,w,heightInpainting,widthInpainting,overlap=10)
        for index,coordinates in enumerate(listCoordinatesSplit):
            #cut in single image of dimension yolo 
            crop_img=im[coordinates[0]:coordinates[0]+heightInpainting,coordinates[1]:coordinates[1]+widthInpainting]                   
            crop_mask=mask[coordinates[0]:coordinates[0]+heightInpainting,coordinates[1]:coordinates[1]+widthInpainting] 
            #check if there are some roi detected
            if(255 in crop_mask):                   
                img_Inpaint = self.Inpaint.doInpainting(crop_img, crop_mask)
                #img_Inpaint = cv2.inpaint(crop_img,cv2.cvtColor(crop_mask,cv2.COLOR_RGB2GRAY),5,cv2.INPAINT_TELEA)

                #cv2.imwrite(pathSaving+"_"+str(index)+"Mask.jpg",img_Inpaint)
                #inpainting
                imageResultInpainting = cv2.cvtColor(img_Inpaint, cv2.COLOR_BGR2RGB) 
                #imageResultInpainting=cv2.copyMakeBorder(imageResultInpainting,4,4,4,4,cv2.BORDER_CONSTANT,value=[0,0,0])                  
            else:
                #inpainting
                imageResultInpainting = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            im_pilInpainting = Image.fromarray(imageResultInpainting)
            newImageInpainting.paste(im_pilInpainting, (coordinates[1], coordinates[0]))

        open_cv_image = np.array(newImageInpainting) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Denoising
        # open_cv_image = cv2.fastNlMeansDenoisingColored(open_cv_image,None,5,5,7,21)

        #stopTotal = time.time()
        #fileLog.write("\nTotal time image: "+os.path.basename(pathImage)+"= "+str(round(stopTotal-startTotal,3))+"s")
        #print("Total time image "+os.path.basename(pathImage)+"= "+str(round(stopTotal-startTotal,3))+"s")
        #cv2.imwrite(pathSaving+"Inpaint.jpg",open_cv_image)              
        return open_cv_image
  
    def run_onfolder(self, input_folder, output_folder): #, filelog="log.txt"):
        #file_log = open(filelog,"a+")
        '''
        dir_name = input_folder.split("/")[-1] # ex Castenaso
        output_folder = os.path.join(output_folder, dir_name)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        '''

        names = os.listdir(input_folder)
        names.sort() 
        for name in names:
            filename_inp = os.path.join(input_folder,name)
            name_without_ext = os. path. splitext(name)[0]
            filename_out_detection = os.path.join(output_folder,name_without_ext+"_Ayolo.jpg")
            filename_out_inpainting = os.path.join(output_folder,name_without_ext+"_Binpaint.jpg")

            #try:
            startTotal = time.time()
            image=cv2.imread(filename_inp,cv2.IMREAD_COLOR) 
            #imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov3(image)
            imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov5(image)
            imgDrawInpaint = self.run_inpainting(image,maskDet )
            stopTotal = time.time()
            time_to_process = str(round(stopTotal-startTotal,3))
            #file_log.write("\nTotal time image: "+filename_inp+"= "+time_to_process+"s")
            print("Total time image "+filename_inp+"= "+time_to_process+"s")
            cv2.imwrite(filename_out_detection,imgDrawDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
            cv2.imwrite(filename_out_inpainting,imgDrawInpaint, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
            #except Exception as e:
            #    print("[Error:] Processing %s" ,e)
                #file_log.write("\n[Error Image not processed]: " + filename_inp)
        #file_log.close()

    def run_onvideo(self, path_video, output_folder): #, filelog="log.txt"):
        cap = cv2.VideoCapture(path_video)
        while(True):
            now = datetime.now()
            s = now.strftime("%Y_%d_%H-%M-%S-%f")
            ret, image = cap.read()
            if ret:
                startTotal = time.time()
                #imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov3(image)
                imgDrawDet,imgMaskDet,maskDet = self.run_detection_yolov5(image)
                imgDrawInpaint = self.run_inpainting(image,maskDet )
                stopTotal = time.time()
                time_to_process = str(round(stopTotal-startTotal,3))
                print("Total time image "+s+"= "+time_to_process+"s")
                #cv2.imwrite(output_folder+"/"+s+"yolo.png",imgDrawDet, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
                cv2.imwrite(output_folder+"/"+s+"inpaint.png",imgDrawInpaint, [int(cv2.IMWRITE_JPEG_QUALITY), 50])    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # install python3.6 https://askubuntu.com/questions/1231543/problem-with-creating-python-3-6-virtual-environment-on-ubuntu-20-04
    input_folder =  os.path.join("..","images","input","febbraio2022")
    output_folder =  os.path.join("..","images","results")
    pr = peopleRemove()
    pr.run_onfolder(input_folder=input_folder, output_folder=output_folder)
    #pr.run_onvideo(path_video ="/home/manuel/visiont3lab-github/public/ai_library/storage/videos/rome-1920-1080-10fps.mp4",output_folder=output_folder)

