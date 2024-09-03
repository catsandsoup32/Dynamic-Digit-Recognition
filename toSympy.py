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
testList4 = ['5','sin', '(', '3', '0', ')', '5', '10']
testList5 = ['5', '(', '5', '+', '3', ')']
testList6 = ['5','cos','(','7',('^','2'),')']
testList7 = ['5',('^', '2')]
testList8 = ['8','+','5','/']
testList9 = ['5', '+','5','8','e',('^','5'),'3',('^','x'),'/','e', ('^','2'), ('^','3')]
testList10 = ['5', '/', '(','5', ('^','5'),('^','x'),')']
testList11 = ['(','10', ')','+','10','/','(','5','+','10',')','S','+','A']
testList12 = ['(', 'x', '+', 'y', ')', 'forward_slash', '3']
testList13 = ['3', 'forward_slash', '(', '4', '+', '8', ')']
testList14 = ['y', '(', '5', ')', 'forward_slash', '5']


# THIS CONVERTS TO LATEX. NOT EQUATION SOLVER
def list_to_sympy(lst):
    expression = rf""
    modifying = False # allows to skip over exponent or subscript
    restartIdx = -1 
    for idx, itm in enumerate(lst):    
        if idx > restartIdx:
            if isinstance(itm, tuple): # exponent or subscript Or decimal

                if not modifying and lst[idx-1] != 'log': # if modifying is False
                    modifier = itm[0]
                    next_items = ""

                    for nextItm in lst[idx+1:len(lst)]:
                        if isinstance(nextItm, tuple):
                            next_items += nextItm[1]
                        else:
                            break
                    
                    expression += rf'{modifier}{{{itm[1]}{next_items}}}' 
                    modifying = True # doesnt keep looping

                if idx == len(lst)-1:
                    return expression
                    
            else: # not a tuple
                modifying = False

                if itm == 'forward_slash':
                    # case singular fraction and case enclosed
                    divideEnclosedStart = True if lst[idx-1] == ')' else False
                    if divideEnclosedStart: 
                        for i in range(idx-1,-1, -1):
                            if i == 0:
                                divideStartIdx = i
                                break
                            if lst[i] == '(' and lst[i-1] != '(':
                                divideStartIdx = i+1
                                break 
                        dividePrevStr = list_to_sympy(lst[divideStartIdx:idx-1])
                    else: 
                        for i in range(idx-1,-1, -1):
                            if i == 0:
                                divideStartIdx = i
                                break
                            elif lst[i] == '+' or lst[i] == '-' or lst[i] == '*':
                                divideStartIdx = i+1 
                                break
                        dividePrevStr = list_to_sympy(lst[divideStartIdx:idx])

                    divideEnclosedEnd = True if lst[idx+1] == '(' else False
                    if divideEnclosedEnd:
                        for i in range(idx+1, len(lst)):
                            if i == len(lst)-1:
                                divideEndIdx = i # COULD bug HERE
                                break
                            if lst[i] == ')' and lst[i+1] != ')':
                                divideEndIdx = i
                                break
                        dividePostStr = list_to_sympy(lst[idx+2: divideEndIdx])
                    else:
                        for i in range(idx+1, len(lst)): # and here
                            if i == len(lst)-1:
                                divideEndIdx = i+1
                                break
                            elif lst[i] == '+' or lst[i] == '-' or lst[i] == '-':
                                divideEndIdx = i+1
                                break
                        dividePostStr = list_to_sympy(lst[idx+1: divideEndIdx])

                    restartIdx = divideEndIdx
                    if divideStartIdx == 0:
                        expression = ''
                    else:
                        if divideEnclosedStart:
                            expression = expression[:divideStartIdx-1]
                        else:
                            expression = expression[:divideStartIdx+1]
                    
                    print(dividePrevStr, dividePostStr)
                    expression = expression + rf"\frac{{{dividePrevStr}}}{{{dividePostStr}}}"
                    if divideEnclosedEnd:
                        if restartIdx == len(lst)-1:
                            return expression
                    else:
                        if restartIdx == len(lst):
                            return expression
    
                elif itm == 'int':
                    pass

                elif itm == 'sigma':
                    pass

                elif itm == 'log':
                    pass
                
                elif itm == 'sin' or itm == 'tan' or itm == 'cos': 
                    '''
                    trigFx = getattr(__import__('sympy'), itm)
                    trigStartIdx = idx + 2
                    trigIdx = idx
                    for nextItm in lst[idx+1:len(lst)]:
                        trigIdx += 1
                        if nextItm == ')':
                            trigEndIdx = trigIdx 
                            break
                    print(f"idxs: {trigStartIdx, trigEndIdx}")
                    evalItems_trig = list_to_sympy(lst[trigStartIdx:trigEndIdx], precision)
                    #expression += str((trigFx(float(evalItems_trig) * pi/180).evalf(precision)))
                    expression += str(trigFx(evalItems_trig))
                    restartIdx = trigEndIdx
                    '''
                    expression += itm
                    
                elif itm == '+' or itm == '-':
                    expression += itm
                elif itm == '*':
                    expression += '\cdot'
                elif itm.isdigit() or itm.isalpha(): 
                    expression += itm
                    '''
                    n = lst[idx+1] if idx < len(lst)-1 else None # fixes reference error
                    if n=='(' or n== 'e' or n=='sin' or n=='tan' or n=='cos' or n=='log' or n=='sigma': 
                        lst.insert(idx+1, 'dot') # implied multiplication
                    '''

                elif itm == '(' or itm == ')':
                    expression += itm
            
                else:
                    pass

                if idx == len(lst)-1: # END CASE 
                    return expression

#print((list_to_sympy(testList14)))





# For log, int, sigma, tan, sin, cos needs special expression
# Need to eval variables and also differentiate from E 

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