import cv2  
import numpy as np
import matplotlib.pyplot as plt

def check_intersect():
    pass
    # generalize to H and V lines

def squareBB(input_image):
    # input_image = cv2.imread(input_image)
    height, width = input_image.shape[0], input_image.shape[1]
    print(f"Image dimensions: {height} (height) x {width} (width)")
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    contours_list, hierarchy = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # retrieval mode tree is for nested contours, chain_simple makes rectangles only four points
    # FOR SOME REASON retr_list doesnt detect some symbols ... 

    print(f"hierarchy: {hierarchy}")

    squareList = [] 
    hLineList = [] # Contains indexes of the minus signs before bb turns into square
    for idx, contour in enumerate(contours_list): # 0 is always the biggest box (entire image)
        x,y,w,h = cv2.boundingRect(contour) # automatically gets rect from many values
        # x and y are bottom left

        if h < 30: # window height / 56? So it scales to the canvas 
            hLineList.append(idx)
        s = max(w, h) # SQUARE SIDE LENGTH

        # centering
        y = y - (w-h)//2 if s == w else y # if width is longer
        x = x - (h-w)//2 if s == h else x   # if height is longer
        squareList.append((x,y,s))
        # cv2.rectangle(binarized, (x,y), (x+s, y+s), (0, 255, 0), 2) # change this later to draw only selected squares

    # RIGHT NOW, getting rid of near-complete overlaps 
    # for holes in numbers
    # Might have to do horizontal and v line logic first
    
    finalSquareList = []
    margin = width/20

    for idx, squareCoord in enumerate(squareList): # DONT DELETE THIS THING pls
        if hierarchy[0][idx][3] == 0:
            finalSquareList.append(squareCoord)
    
    ''' [[[-1 -1  1 -1]
        [ 4 -1  2  0]
        [ 3 -1 -1  1]
        [-1  2 -1  1]
        [-1  1  5  0]
        [ 6 -1 -1  4]
        [-1  5 -1  4]]]'''


    '''
    for kingIdx, squareCoord in enumerate(squareList): 
        xH, yH, sH = squareCoord[0], squareCoord[1], squareCoord[2]
        x2H, y2H = xH + sH, yH - sH 
        print(f"h-coords: {xH, yH, sH}")
        unionAreasList = []
        for peasantIdx, otherSquare in enumerate(squareList): # checks against every other bb
            if kingIdx != peasantIdx:
                xC, yC, sC = otherSquare[0], otherSquare[1], otherSquare[2] # bottom left (x, y), and side length
                x2C, y2C = xC + sC, yC - sC

                if xH > (xC - margin) and yH < (yC + margin) and (xH + sH) < (xC + sC + margin) and (yH - sH) > (yC - sC - margin):
                    
                    # gets the union
                    xU = max(xH, xC)
                    yU = min(yH, yC)
                    x2U = min(xH + sH, xC + sC)
                    y2U = max(yH - sH, yC - sC)

                    wU = x2U - xU
                    hU = yU - y2U

                #print(wU, hU)
                #unionAreasList.append(hU * wU)
        '''
        
            

    for squares in finalSquareList:
        cv2.rectangle(binarized, (squares[0],squares[1]), (squares[0]+squares[2], squares[1]+squares[2]), (0, 255, 0), 2) 

        
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

    plt.imshow(binarized)
    plt.show()

    return finalSquareList
 

# squareBB('rect_test3.png')



