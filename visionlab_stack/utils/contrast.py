import cv2
import os
import numpy as np

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf



class control:
    def __init__(self):
        self.title_window = 'Contrast-Brightness'
        cv2.namedWindow(self.title_window,cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Contrast", self.title_window, 0, 100, self.on_contrast)
        cv2.createTrackbar("Brightness", self.title_window, 0, 100,self.on_brightness)
        #cv2.createTrackbar("Lab", self.title_window, 1, 100,self.on_test)
        fp = "/home/manuel/visiont3lab-github/public/people-remove/images/input/Molino-Ariani/s_111_c_114-20201219_094253M.jpg"
        self.frame = cv2.imread(fp,1)
        self.brightness = 50
        self.contrast =50
        self.on_brightness(self.brightness)
        self.on_brightness(self.contrast)
        
    def on_brightness(self,brightness):
        self.brightness = brightness
        adjusted = apply_brightness_contrast(self.frame,self.brightness-50,self.contrast-50)
        cv2.imshow(self.title_window, adjusted)
    def on_contrast(self,contrast):
        self.contrast = contrast
        adjusted = apply_brightness_contrast(self.frame,self.brightness-50,self.contrast-50)
        cv2.imshow(self.title_window, adjusted)
    def on_test(self, val):
        imLab = cv2.cvtColor(self.frame,cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=val/10, tileGridSize=(8,8))
        imLab[:,:,0] = clahe.apply(imLab[:,:,0])
        adjusted = cv2.cvtColor(imLab,cv2.COLOR_LAB2RGB)
        cv2.imshow(self.title_window, adjusted)

# Show some stuff
c = control()
#c.on_test(0)

# Wait until user press some key
cv2.waitKey()