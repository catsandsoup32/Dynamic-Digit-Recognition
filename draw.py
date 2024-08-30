from tkinter import Tk, Canvas, Button, Scale, HORIZONTAL, RAISED, SUNKEN, ROUND, TRUE # dont use wildcard import
from tkinter.colorchooser import askcolor
import threading

from PIL import Image, ImageGrab
import PIL.ImageOps    
import pyautogui
import pygetwindow as gw
import io 
import numpy as np
import matplotlib.pyplot as plt

import mss
from bounding_box_cap import squareBB
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from train import transform, MNIST_transform
from models import CNN13, VamsiNN, CNN14, CNN_9
from dataloader import class_Labels_Length


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self, model, model_folder, transform):
        self.root = Tk()
        self.root.title("Paint Window")
        self.window_title = "Paint Window"
        self.model = model
        self.model_folder = model_folder
        self.transform = transform

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.clear_button = Button(self.root, text='clear', command=self.clear_canvas)
        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.choose_size_button = Scale(self.root, from_=5, to=50, orient=HORIZONTAL)

        self.pen_button.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.eraser_button.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.clear_button.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.predict_button.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.choose_size_button.grid(row=0, column=4, padx=5, pady=5, sticky='w')
        
        # PYTORCH STUFF HERE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_folder, map_location=self.device, weights_only=True))
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

        # Canvas and output area
        self.c = Canvas(self.root, bg='white')
        self.output_frame = Canvas(self.root, bg='lightgrey')  # Placeholder for output area

        # Arrange canvas and output area in a 2-column layout
        self.c.grid(row=1, column=0, sticky='nsew', padx=0, pady=0)
        self.output_frame.grid(row=1, column=1, sticky='nsew', padx=0, pady=0)
        
        # Configure grid to adjust sizes
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=3)  
        self.root.grid_columnconfigure(1, weight=1)

        self.root.attributes('-fullscreen', True)  # Fullscreen mode
        self.setup()
        self.root.bind("<Escape>", self.exit_fullscreen)  # Bind Escape to exit fullscreen
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Q>', self.close_window)
        self.root.after(100, self.print_window_size)
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.objects = []
        self.screenshot_img = None

    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)
        self.root.quit()

    def use_pen(self):
        self.activate_button(self.pen_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def clear_canvas(self):
        self.c.delete("all")
        self.objects.clear()  
        
    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = "white" if self.eraser_on else (self.color if self.color else self.DEFAULT_COLOR)  
        if self.old_x and self.old_y:
            obj_id = self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.objects.append(obj_id)  # Store the object's ID for undo
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def undo(self, event=None):
        if self.objects:
            # Remove the last object added to the canvas
            last_object = self.objects[max(0,len(self.objects)-50):len(self.objects)]
            for item in last_object:
                self.c.delete(item)
            self.objects = self.objects[0: len(self.objects)-50]

    def get_window_size(self):
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        return width, height

    def print_window_size(self):
        size = self.get_window_size()
        print(f"Window size: {size}")

    def close_window(self, event=None):
        self.root.destroy()

    def bounding_box(self, x, y, width, height, color):
        self.c.create_line(x,y,x,y+height, fill="green", width=5)
        self.c.create_line(x,y+height,x+width,y+height, fill="green", width=5)
        self.c.create_line(x+width,y+height,x+width,y, fill="green", width=5)
        self.c.create_line(x+width, y, x, y, fill="green", width=5)


    def predict(self):
        # Hide the button grid
        self.pen_button.grid_forget()
        self.eraser_button.grid_forget()
        self.clear_button.grid_forget()
        self.predict_button.grid_forget()
        self.choose_size_button.grid_forget()

        self.root.update()  # Force update
        self.root.update_idletasks()  # Ensure all events are processed

        left = self.c.winfo_rootx() # these are TOP LEFT
        top = self.c.winfo_rooty()  
        width = self.c.winfo_width()
        height = self.c.winfo_height() 
        
        with mss.mss() as sct:
            window = {"top": top, "left": left, "width": width, "height": height}
            entire_ss = sct.grab(window) 

            imageBGR = cv2.cvtColor(np.array(entire_ss), cv2.COLOR_BGRA2BGR) # np array and bgr for cv2
            bbList = squareBB(imageBGR) # list of bounding boxes
        
        bb_ss_list = []
        for box in bbList: 
            # draws bounding boxes, COMMENT OUT FOR ANY DEMO
            x = box[0]
            y = box[1]
            side = box[2]
            self.bounding_box(x=x, y=y, width=side, height=side, color='green') # calls function

            with mss.mss() as sct: # FIX THESE COORDINATES
                bb_window = {"top": y, "left": x, "width": side, "height": side} # opencv BOTTOM LEFT
                bb_ss = sct.grab(bb_window)
                
            bb_ss_list.append(bb_ss)
        
        input_image = cv2.cvtColor(np.array(bb_ss_list[0]), cv2.COLOR_BGRA2BGR)
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        plt.imshow(gray_image, cmap='gray')
        plt.show()

        # Show the button grid again
        self.pen_button.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.eraser_button.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.clear_button.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.predict_button.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        self.choose_size_button.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        self.run_inference(symList = bb_ss_list)

    def run_inference(self, symList):
        
        for syms in symList: # these are mss screenshots
            pil_image = Image.fromarray(np.array(syms))
            ss_img = pil_image.resize((45,45)) # NEVER INVERT THIS, trained on B-on-W data
            input_tensor = self.transform(ss_img)

            if isinstance(input_tensor, torch.Tensor):   
                showIm = np.squeeze(input_tensor.numpy()) 
                plt.imshow(showIm)
                plt.show() 

                input_tensor = torch.unsqueeze(input_tensor, 0) # Add batch dim
                torch.set_printoptions(threshold=1000, edgeitems=10)
                #print(f"Image tensor of size {input_tensor.size()}: {input_tensor}")
                input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.softmax(self.model(input_tensor))
            
            output = output.squeeze(0)  # Remove batch dimension
            predictions = output.cpu().detach().numpy() if torch.cuda.is_available() else output
            predictions = list(predictions) 
            
            labels_df = class_Labels_Length('data/extracted_images') # imported from dataloader
            classes = labels_df['Class_Names'].tolist()
            
            max_index = predictions.index(max(predictions))  # Get the index of the highest prediction
            max_class_name = classes[max_index]
            
            plt.clf()
            plt.bar(classes, predictions, color = 'skyblue')
            plt.annotate(f'Highest: {max_class_name}',
                xy=(max_index, predictions[max_index]),
                xytext=(max_index, predictions[max_index] + 0.1),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=1.5))
            plt.show()
        

if __name__ == '__main__':
    paint_app = Paint(model=CNN_9(), model_folder='save_states/CNNmodel9Epoch20.pt', transform=transform)

# CNN 9 actually seems to work the best ?? 
# with pen size 5 when drawing small; pen size is important (could use image processing to scale down)
