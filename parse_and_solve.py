import sympy 
from sympy import evalf
from sympy.parsing.latex import parse_latex
from toSympy import list_to_sympy
from sympy import pi
from sympy import Float
import re

test1 = r"\log(\e)"

# Convert LaTeX to SymPy
def solver(latex, varDict):
    if varDict is not None:
        for key, value in varDict.items():
            if key in latex:
                latex = latex.replace(key, list_to_sympy(value))    

    latex = latex.replace(r'\pi', '3.1415926') if r'\pi' in latex else latex
    latex = latex.replace(r'\e', '2.7182818') if r'\e' in latex else latex

    sympy_expr = parse_latex(latex) # built in function
    num_eval = sympy_expr.evalf(3)
    if isinstance(num_eval, Float): # float
        print('float')
        return_print = str(int(sympy_expr.evalf(3) * 1000)/1000) # rounds
        return return_print[:-2] if return_print[-2:len(return_print)] == '.0' else return_print
    else:
        print('symbols')
        return str(sympy_expr.evalf())


