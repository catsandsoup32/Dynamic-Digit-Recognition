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
testList15 = ['sin', '(', 'pi',')']
testList16 = ['infty']
testList17 = ['int', '5', 'x','d','x']
testList18 = ['log',('_','1'),('_','0'),'+','10']
testList19 = ['log','10']
testList20 = ['log', ('_', '1'), ('_', '0'), '1', '0', '0']



# THIS CONVERTS TO LATEX. NOT EQUATION SOLVER
def list_to_sympy(lst):
    expression = rf""
    modifying = False # allows to skip over exponent or subscript
    restartIdx = -1 
    for idx, itm in enumerate(lst):    
        if idx > restartIdx:
            if isinstance(itm, tuple): # exponent or subscript Or decimal

                if itm[1] == 'dot':
                    expression += '.'

                elif not modifying and lst[idx-1] != 'log': # if modifying is False
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
                        dividePrevStr = list_to_sympy(lst[divideStartIdx:idx])
                    else: 
                        for i in range(idx-1,-1, -1):
                            if i == 0:
                                divideStartIdx = i
                                break
                            elif lst[i] == '+' or lst[i] == '-' or lst[i] == '*' or lst[i] == '(':
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
                            elif lst[i] == '+' or lst[i] == '-' or lst[i] == '-' or lst[i] == ')':
                                divideEndIdx = i+1
                                break
                        dividePostStr = list_to_sympy(lst[idx+1: divideEndIdx])

                    restartIdx = divideEndIdx
                    if divideStartIdx == 0:
                        expression = ''
                    else:
                        if divideEnclosedStart:
                            expression = expression[:divideStartIdx]
                        else:
                            expression = expression[:divideStartIdx]
                    
                    print(dividePrevStr, dividePostStr)
                    expression = expression + rf"\frac{{{dividePrevStr}}}{{{dividePostStr}}}"
                    if divideEnclosedEnd:
                        if restartIdx == len(lst)-1:
                            return expression
                    else:
                        if restartIdx == len(lst):
                            return expression
                
                elif itm == 'infty':
                    expression += '\\' + itm
    
                elif itm == 'int':
                    if isinstance(lst[idx+1], tuple) or isinstance(lst[idx+2], tuple): # bounds exist
                        lowerBound = lst[idx+1][1] if isinstance(lst[idx+1], tuple) else lst[idx+1]
                        upperBound = lst[idx+2][1] if isinstance(lst[idx+2], tuple) else lst[idx+2]
                        expression += '\\'+itm+rf'_{{{lowerBound}}}^{{{upperBound}}}'
                        intStartIdx = idx + 3 
                    else:
                        intStartIdx = idx + 1 
                        expression += '\\'+itm

                    for dIdx, dItm in enumerate(lst[idx+1:len(lst)]):
                        if dItm == 'd':
                            intStopIdx = dIdx + 2 # the index after variable 
                            break
                        
                    expression += list_to_sympy(lst[intStartIdx:intStopIdx-1]) + f'\\,d' 
                    restartIdx = intStopIdx-1

                elif itm == 'sigma' or itm == 'theta' or itm == 'phi' or itm == 'beta' or itm == 'alpha' or itm == 'delta' or itm == 'gamma' or itm == 'lambda':
                    expression += '\\' + itm

                # testList20 = ['log', ('_', '1'), ('_', '0'), '1', '0', '0']
                elif itm == 'log':
                    expression += '\\' + itm 
                    base = True if isinstance(lst[idx+1], tuple) else False
                    if base:
                        expression += r'_{'
                        for subIdx, subItm in enumerate(lst[idx+1:len(lst)]): 
                            if isinstance(subItm, tuple):
                                expression += subItm[1] 
                            else:
                                expression += r'}'
                                restartIdx = subIdx 
                                break
                    else: 
                        expression += r'_{' + '\\' + r'e}'

                elif itm == 'pi':
                    expression += '\\' + itm
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
                    expression += '\\' + itm # for latex format
                    
                elif itm == '+' or itm == '-':
                    expression += itm
                elif itm == 'dot':
                    expression += '\cdot'
                elif itm == 'e':
                    expression += '\\' + itm
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

#print(list_to_sympy(testList20))

# For log, int, sigma, tan, sin, cos needs special expression
# Need to eval variables and also differentiate from E 

#function = getattr(__import__('sympy'), 'sin')
#print(str(function(pi/6)))
