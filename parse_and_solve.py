import sympy 
from sympy import evalf
from sympy.parsing.latex import parse_latex
from toSympy import list_to_sympy
from sympy import pi


test1 = r"\frac{1}{2} + 5^{2}"

# Convert LaTeX to SymPy
def solver(latex, varDict):
    for key, value in varDict.items():
        if key in latex:
            latex = latex.replace(key, list_to_sympy(value))    
    sympy_expr = parse_latex(latex) # built in function
    return str(sympy_expr.evalf(3))

#print(solver('\sin(3.14158)'))
