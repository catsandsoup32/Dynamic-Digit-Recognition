import cv2  
import numpy as np
import matplotlib.pyplot as plt

x_margin = 40

def squareBB(input_image):
    #input_image = cv2.imread(input_image) # comment this out when using draw.py
    #height, width = input_image.shape[0], input_image.shape[1]
    #print(f"Image dimensions: {height} (height) x {width} (width)")
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    contours_list, hierarchy = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # retrieval mode tree is for nested contours, chain_simple makes rectangles only four points

    squareList = [] 
    hLineList = [] # Contains indexes of the minus signs before bb turns into square
    dotList = []
    for idx, contour in enumerate(contours_list): 
        x,y,w,h = cv2.boundingRect(contour) # automatically gets rect from many values (x, y) bottom left corner
        if h < 30:
            if w > h*2: 
                hLineList.append(idx) # for minuses
            else: 
                dotList.append(idx)
                
        s = max(w, h) # SQUARE SIDE LENGTH
        # centering
        y = y - (w-h)//2 if s == w else y # if width is longer
        x = x - (h-w)//2 if s == h else x   # if height is longer
        squareList.append([x,y,s,idx]) # put original width and height here !!!!    

    # THIS FIXES HOLES
    for idx, squareCoord in enumerate(squareList): 
        if hierarchy[0][idx][3] == 0: 
            pass
        else: 
            squareCoord[3] = None
    
    print(f"hLine original: {hLineList}")
    # check minuses against each other to fix equals signs
    if len(hLineList) >= 2:
        for minusIdx in hLineList:
            x1, y1 = squareList[minusIdx][0], squareList[minusIdx][1]
            outBreak = False
            for minusIdx2 in hLineList:
                x2, y2 = squareList[minusIdx2][0], squareList[minusIdx2][1]
                if minusIdx != minusIdx2 and abs(x1-x2) < x_margin:
                
                    # remove equal sign indices
                    hLineList.remove(minusIdx)
                    hLineList.remove(minusIdx2)

                    for values in squareList:
                        values[1] = (y1+y2)//2 if values[3] == minusIdx else values[1] 
                        values[3] = None if values[3] == minusIdx2 else values[3]  
                                        
                    outBreak = True
                    break
            if outBreak:
                break

    print(f"hLine removed: {hLineList}")
        
    # combines i's, j's, etc. and also crops dots
    for dotIdx in dotList:
        x, y, s = squareList[dotIdx][0], squareList[dotIdx][1], squareList[dotIdx][2]
        trueDot = True
        for squares in squareList:
            if squares[3] != dotIdx and abs(squares[0] + squares[2]//2 - x) < x_margin and squares[3] != None:
                x0, y0, s0 = squares[0], squares[1], squares[2]
                d = squares[1] - y
                squares[2] = s0 + d
                squares[0] = x0 - d//2
                squares[1] = y0 - d
                squareList[dotIdx][3] = None
                trueDot = False
                break
            
        #if trueDot:
            #centerX = x + s//2
            #centerY = y - s//2
            #squareList[dotIdx][0], squareList[dotIdx][1], squareList[dotIdx][2] = centerX-5, centerY+5, 10
 

    finalSquareList = []
    xCoordList = []
    sideList = []
    for squares in squareList:
        if squares[3] is not None:  # gets all valid bounding boxes
            finalSquareList.append(squares)  
            xCoordList.append(squares[0])
            sideList.append(squares[2])
    
    sideList.sort()
    for squares in finalSquareList: # gets square with largest side
        if squares[2] == sideList[-1]:
            largestSquare = squares
            break
            
    xCoordList.sort()
    sortedFinalSquareList = []
    insertIdx = 0

    '''
    # APPEND A 'NUM' or 'DEN' term  
    # only check minuses that are near center to allow for other minuses in num / denom
    centerLine = largestSquare[1] + largestSquare[2]//2 # top + side/2

    hLineCompleteList = []
    for hLineIdx in hLineList:
        for squares in finalSquareList:
            if hLineIdx == squares[3]:
                hLineCompleteList.append((squares[0], squares[1], squares[2], squares[3])) 

    print(f"HlinewithY: {hLineCompleteList}")

    for idx, squares in enumerate(finalSquareList):
        for hTuple in hLineCompleteList:
            if squares[3] != hTuple[3]: # a horizontal line won't check itself
                centerX = squares[0] + squares[2]//2
                centerY = squares[1] + squares[2]//2
                if centerX >= hTuple[0] and centerX < hTuple[0] + hTuple[2] and centerY >= hTuple[1] and centerY <= hTuple[1] + hTuple[2]: 
                 # if the center of the symbol is within minus bb
                    hCenter = hTuple[1] + hTuple[2]//2
                    if centerY < hCenter: # above
                        finalSquareList[idx].append('num')
                        print(hTuple[3])
                    elif centerY > hCenter:
                        finalSquareList[idx].append('den')
    
    print(finalSquareList)
    '''

    while insertIdx < len(finalSquareList):
        for squares in finalSquareList:
            if squares[0] == xCoordList[insertIdx]: 
                sortedFinalSquareList.append(squares)
                insertIdx += 1
                break

    for squares in sortedFinalSquareList:
        cv2.rectangle(binarized, (squares[0],squares[1]), (squares[0]+squares[2], squares[1]+squares[2]), (0, 255, 0), 2) 

    #plt.imshow(binarized)
    #plt.show()

    #print(sortedFinalSquareList)
    return (sortedFinalSquareList, largestSquare)
 
#squareBB('equals_test.png')



