import sympy 
from sympy import evalf
from sympy.parsing.latex import parse_latex


latex_expr = r"\frac{1}{2} + 5^{2}"

# Convert LaTeX to SymPy
sympy_expr = parse_latex(latex_expr)

print(sympy_expr.evalf(10))  # Output: 1/2 + x**2

# Solver