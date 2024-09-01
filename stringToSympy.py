import sympy 
from sympy import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)  # Enable LaTeX rendering
import tkinter as tk
from PIL import Image, ImageTk
import io

init_printing(use_unicode=True)
x = symbols('x')
y = symbols('y')

test = latex(Integral(x+y, x))


print(test)

'''
class LaTeXCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Display LaTeX in Tkinter")
        
        self.canvas = tk.Canvas(root, width=600, height=400, bg="white")
        self.canvas.pack()

        # Entry widget for LaTeX input
        self.latex_entry = tk.Entry(root, width=50)
        self.latex_entry.pack()
        self.latex_entry.insert(0, r"$\frac{a}{b}$")

        # Button to render LaTeX
        self.render_button = tk.Button(root, text="Render LaTeX", command=self.render_latex)
        self.render_button.pack()

    def render_latex(self):
        latex_code = self.latex_entry.get()

        # Create a Matplotlib figure
        fig = Figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, latex_code, fontsize=20, ha='center', va='center')
        ax.axis('off')

        # Convert the Matplotlib figure to a PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.tk_image = ImageTk.PhotoImage(image)

        # Display the image on the Tkinter canvas
        self.canvas.create_image(300, 200, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = LaTeXCanvasApp(root)
    root.mainloop()
'''