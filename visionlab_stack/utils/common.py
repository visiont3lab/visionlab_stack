
import numpy as np
import imutils
import cv2
import time

def increaseRoi(yoloCoordinates,percentage):
    # increase every dimension of percentage of it
    # W=100 became 110 with a  percentage of 10% 
    yoloCoordinates[0]=round(yoloCoordinates[0]-(yoloCoordinates[2]*percentage)/2)
    yoloCoordinates[1]=round(yoloCoordinates[1]-(yoloCoordinates[3]*percentage)/2)
    yoloCoordinates[2]=round(yoloCoordinates[2]+(yoloCoordinates[2]*percentage))
    yoloCoordinates[3]=round(yoloCoordinates[3]+(yoloCoordinates[3]*percentage))
    return yoloCoordinates

def createImageHole(image,x,y,x_plus_w,y_plus_h):
    color = (255,255,255) #self.COLORS[class_id]
    cv2.rectangle(image, (x,y), (x_plus_w,y_plus_h), color, thickness=-1)

def createMask(mask,x,y,x_plus_w,y_plus_h):
    thickness = -1
    color = (255,255,255)
    mask = cv2.rectangle(mask, (x,y),(x_plus_w,y_plus_h), color, thickness) 

def splitImage(imHeight,imWidth,heightYolo,widthYolo,overlap):   
    # height, width, number of channels in image
    imgheight = imHeight
    imgwidth = imWidth   
    #newImage = Image.new('RGB', (imgwidth, imgheight))
    start = time.time()
    listCoordinates=[]
    for i in range(0,imgheight,heightYolo-overlap):
            for j in range(0,imgwidth,widthYolo-overlap): 
                '''    
                #single cropped image    
                if(j==0 and i==0):
                    crop_img = im[i:i+heightYolo,j:j+widthYolo] 
                    #print(i,i+heightYolo,j,j+widthYolo)
                elif(i==0 and j!=0):
                   crop_img = im[i:i+heightYolo,j:j+widthYolo] 
                   #print(i,i+heightYolo,j,j+widthYolo)
                elif(i!=0 and j==0):
                    crop_img = im[i:i+heightYolo,j:j+widthYolo] 
                    #print(i,i+heightYolo,j,j+widthYolo)
                else:
                  crop_img = im[i:i+heightYolo,j:j+widthYolo]
                  #print(i,i+heightYolo,j,j+widthYolo) 
                '''
                listCoordinates.append((i,j))
                              
    stop = time.time()    
    #print("Total time=",stop-start,"s")       
    return listCoordinates

def pyramid(image, scale=2, minSize=(100, 100)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

def splitImages(im):
    # 3*3 = 9  square
    num=3
    width =  416 #int(im.shape[0]/num)
    height = 416 #int(im.shape[1]/num)

    imgwidth =  im.shape[0]
    imgheight = im.shape[1]
    
    imgs = []
    for j in range(0,imgheight,height):   # row
        for i in range(0,imgwidth,width): # col

            imCrop = im[i:i+width,j:j+height]
            imgs.append(imCrop)
            #cv2.rectangle(im,(j,i),(j+height,i+width), [0,255,255], 2)
            
            if ((j+height)<imgheight):
                overlapImCropH = im[i:i+width,j+int(0.5*height):j+int(1.5*height)]
                imgs.append(overlapImCropH)
                #cv2.imshow("overlapImCropH", overlapImCropH)
            if ((i+width)<imgwidth):
                overlapImCropW = im[i+int(0.5*width):i+int(1.5*width),j:j+height]
                imgs.append(overlapImCropW)
                #cv2.imshow("overlapImCropW", overlapImCropW)
                #cv2.rectangle(im,(j+int(0.5*width),i),(j+int(1.5*width),i+height), [0,255,0], 2)
            #cv2.imshow("ims", im)
            #cv2.imshow("imCrop", imCrop)    
            #cv2.waitKey(0)
    return imgs

