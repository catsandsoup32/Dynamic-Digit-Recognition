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

# STORE VALUES IN VARIABLES GLOBAL
# use r"" for raw string


testList = ['(', '5', '+', '3', ')', 'dot', '3']
testList2 = ['log', ('_', '5'), '(', '10', ')', '=']
testList3 = ['v', 'dot', 'r', '+', '0', '=']
testList4 = ['sin', '(', '3', '0', ')']



def list_to_sympy(lst):
    outputMode = 'normal' # output can be norm for only add subtract etc. but fx for functions
    expression = rf""
    operations = {
        '+': '+',
        '-': '-',
        'dot': '*',
        '/': '/'
    }

    modifying = False # allows to skip over exponent or subscript
    for idx, itm in enumerate(lst):
        #print(expression)

        if isinstance(itm, tuple): # exponent or subscript
            if not modifying and lst[idx-1] != 'log': # if modifying is False
                modifier = itm[0]
                next_items = ""

                for nextItm in lst[idx:len(lst)]:
                    if isinstance(nextItm, tuple):
                        next_items += nextItm[1]
                    else:
                        break
            
                expression += rf'{{{modifier}{next_items}}}' 
                modifying = True # doesnt keep looping
                
        else: 
            modifying = False

            if itm == '=':
                expression += itm
            
            elif itm == 'int':
                pass

            elif itm == 'sigma':
                pass

            elif itm == 'log':
                pass
            
            elif itm == 'sin' or itm == 'tan' or itm == 'cos': 
                trigFx = getattr(__import__('sympy'), itm)
                for nextItm in lst[idx+1:len(lst)]:
                    if nextItm == ')':
                        endIdxTrig = idx # this is the index to start again now
                        break
                evalItems_trig = (", ".join(lst[idx+2:endIdxTrig]))
                expression += trigFx(evalItems_trig)

            elif itm in operations:
                expression += operations[itm]

            elif itm.isdigit() or itm.isalpha() or itm == '(' or itm == ')':
                expression += itm
        
            else:
                pass

            if idx == len(lst)-1: # END CASE 
                return expression

#print(list_to_sympy(testList))

# DO SPLIT

# print(sin(30).evalf())  times pi/180 every time with

# For a subscript, must be in format _{...}
# For an exponent, ^{...}
# For log, int, sigma, tan, sin, cos needs special expression

#function = getattr(__import__('sympy'), 'sin')
#print(str(function(pi/6)))


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