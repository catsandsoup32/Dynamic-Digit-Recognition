from PIL import Image, ImageDraw, ImageTk
import torch
import torchvision.transforms as transforms

from draw import Paint
drawCanvas = Paint()

drawCanvas.predict()

# Access the screenshot image
draw_img = drawCanvas.screenshot_img

