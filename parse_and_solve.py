import sympy 
from sympy import evalf
from sympy.parsing.latex import parse_latex
from sympy import pi


test1 = r"\frac{1}{2} + 5^{2}"

# Convert LaTeX to SymPy
def solver(latex):
    sympy_expr = parse_latex(latex)
    return str(sympy_expr.evalf(3))

print(solver('\sin(3.14158)'))
