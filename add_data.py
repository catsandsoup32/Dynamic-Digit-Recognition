import os
import PIL 
import matplotlib.pyplot as mpl
import numpy as np
from PIL import Image, ImageDraw


img = Image.new('L',(45, 45), 0)

for i in range(1000,2000, 1):
    #img.save(os.path.join('data/extracted_images_new/dot', f'image_{i}.jpg'))
    pass

'''
counter = -1
for root, dirs, files in os.walk('bhmsds/symbols'):
    for f in files:
        counter += 1
        file_path = os.path.join('bhmsds/symbols', f)
        if file_path[15:20] == 'slash':
            image = Image.open(file_path)
            image.resize((45, 45))
            image.save(os.path.join('data/extracted_images_new/forward_slash', f'BHMSDS_slash_{counter}.jpg'))            
'''      
# deleted this forward slash bhmsds, still can use this function in case need to remove










