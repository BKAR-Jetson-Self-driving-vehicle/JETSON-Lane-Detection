import os
import glob
import time

import cv2
from cv2 import COLOR_BAYER_BG2BGR
import numpy as np

cap = cv2.VideoCapture('./data/tokyo.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

def EdgeDetect(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

    frame1 = np.concatenate((sobelx, sobely), axis=1)
    frame2 = np.concatenate((edges, sobelxy), axis=1)
    return np.concatenate((frame1, frame2), axis=0)

prev_frame_time = 0
new_frame_time = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
    
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, threshold_img = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        result = EdgeDetect(frame)

        cv2.putText(result, "FPS: "+fps, (7, 40), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('Video', cv2.resize(result, (1280, 720), interpolation = cv2.INTER_AREA))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
