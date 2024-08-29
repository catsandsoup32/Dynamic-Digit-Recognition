import cv2  
import numpy as np
import matplotlib.pyplot as plt

def check_intersect():
    pass
    # generalize to H and V lines

def squareBB(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    contours_list, _ = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # retrieval mode tree is for nested contours, chain_simple makes rectangles only four points
    # FOR SOME REASON retr_list doesnt detect some symbols ... 

    squareList = [] 
    hLineList = [] # Contains indexes of the minus signs before bb turns into square
    for idx, contour in enumerate(contours_list[1: len(contours_list)]): # 0 is always the biggest box (entire image)
        x,y,w,h = cv2.boundingRect(contour) # automatically gets rect from many values
        # x and y are bottom left

        if h < 30: # window height / 56? So it scales to the canvas 
            hLineList.append(idx)
        s = max(w, h) # SQUARE SIDE LENGTH

        # centering
        y = y - (w-h)//2 if s == w else y # if width is longer
        x = x - (h-w)//2 if s == h else x   # if height is longer
        squareList.append((x,y,s))
        cv2.rectangle(binarized, (x,y), (x+s, y+s), (0, 255, 0), 2) # change this later to draw only selected squares


    # check ONLY MINUSES with everything else
    for index in hLineList: # hLineList contains the INDEXES of SQUARELIST that are minuses
        minusCoords = squareList[index]
        
        leftIntersect = False
        rightIntersect = False
        topIntersect = False
        downIntersect = False

        for squareIdx, square in enumerate(squareList): 
            if index != squareIdx: # so it doesn't check itself
                # check if intersects
                if (square[0] + square[2]) - minusCoords[0] > 0 and (square[0] + square[2]) - minusCoords[0] < 20:
                    pass
                
    # Check for similar X values above and below. 
    # If none, then it is a minus sign
    # If it overlaps, get rid of it    
    # If it detects one other small area, then these two become an equal sign
    # If it detects two big areas, then it is a division sign

    #plt.imshow(binarized)
    #plt.show()

    return squareList
 

# squareBB()

# get bounding box from opencv
# input image is the ENTIRE TKINTER CANVAS


