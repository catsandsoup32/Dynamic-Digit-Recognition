from tkinter import *
from tkinter.colorchooser import askcolor

from PIL import Image, ImageGrab
import pyautogui
import pygetwindow as gw
import io 
import numpy as np
import matplotlib.pyplot as plt
    
class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()
        self.root.title("Paint Window")
        self.window_title = "Paint Window"

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        #self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        #self.brush_button.grid(row=0, column=1)

        #self.color_button = Button(self.root, text='color', command=self.choose_color)
        #self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.clear_button = Button(self.root, text='clear', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=2)

        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=1)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.bind('<Control-z>', self.undo)
        self.root.bind('<Escape>', self.close_window)
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

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

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
        paint_color = 'white' if self.eraser_on else self.color
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

    def predict(self):
        self.root.update()  # Force update
        self.root.update_idletasks()  # Ensure all events are processed
        # Get the window by its title
        window = gw.getWindowsWithTitle(self.window_title)[0]
        if not window:
            raise ValueError(f"No window with title '{self.window_title}' found.")
        
        left = self.root.winfo_rootx()
        top = self.root.winfo_rooty()
        right = left + self.root.winfo_width()
        bottom = top + self.root.winfo_height()

        bbox = (left, top, right, bottom)
        print(bbox)

        # Capture the area of the screen where the window is located
        img = pyautogui.screenshot(region=bbox)

        plt.imshow(img)
        plt.axis('off') 
        plt.show()
       

if __name__ == '__main__':
    # paint_app = Paint()
    pass