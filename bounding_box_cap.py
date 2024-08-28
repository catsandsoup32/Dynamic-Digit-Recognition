import cv2  
import numpy as np
import matplotlib.pyplot as plt

def detect_contours():
    input_image = cv2.imread('opencv_bb_test_spis.png', 0)
    _, binarized = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)

    contours_list, _ = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # retrieval mode tree is for nested contours, chain_simple makes rectangles only four points

    coordinatesList = []
    for idx, c in enumerate(contours_list[1: len(contours_list)]): # 0 is always the biggest one around entire image
        contour = c
        x,y,w,h = cv2.boundingRect(contour) # automatically gets rect from many values
        

        cv2.rectangle(binarized, (x,y), (x+w, y+h), (0, 255, 0), 2)

    print(coordinatesList)
    plt.imshow(binarized)
    plt.show()
 

detect_contours()



# get bounding box from opencv
# input image is the ENTIRE TKINTER CANVAS


