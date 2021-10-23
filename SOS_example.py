import sympy as sp
import numpy as np
from SumOfSquares import SOSProblem, poly_opt_prob


# """example 1"""

# # Defines symbolic variables and polynomial
# x, y = sp.symbols('x y')
# p = 2*x**4 + 2*x**3*y - x**2*y**2 + 5*y**4
# prob = SOSProblem()

# # Adds Sum-of-Squares constaint and solves problem
# const = prob.add_sos_constraint(p, [x, y])
# prob.solve()

# # Prints Sum-of-Squares decomposition
# print(sum(const.get_sos_decomp()))

# x, y = sp.symbols('x y')
# p = x**4*y**2 + x**2*y**4 - 3*x**2*y**2 + 1
# prob = SOSProblem()
# prob.add_sos_constraint(p, [x, y])
# prob.solve() # Raises SolutionFailure error due to infeasibility

"""example 2"""

x, y, s, t = sp.symbols('x y s t')
p = s*x**6 + t*y**6 - x**4*y**2 - x**2*y**4 - x**4 \
    + 3*x**2*y**2 - y**4 - x**2 - y**2 + 1
prob = SOSProblem()
prob.add_sos_constraint(p, [x, y])
sv, tv = prob.sym_to_var(s), prob.sym_to_var(t)
prob.solve()
prob.set_objective('min', sv+tv)
print(sv.value, tv.value)

x, y, s, t = sp.symbols('x y s t')
p = s*x**6 + t*y**6 - x**4*y**2 - x**2*y**4 - x**4 \
    + 3*x**2*y**2 - y**4 - x**2 - y**2 + 1
prob = SOSProblem()
prob.add_sos_constraint(p, [x, y])
sv, tv = prob.sym_to_var(s), prob.sym_to_var(t)
prob.set_objective('min', sv+tv)
prob.solve()
print(sv.value, tv.value)
# returns s ~ 1.203, t ~ 1.203

x, y, t = sp.symbols('x y t')
p1 = t*(1 + x*y)**2 - x*y + (1 - y)**2
p2 = (1 - x*y)**2 + x*y + t*(1 + y)**2
prob = SOSProblem()
prob.add_sos_constraint(p1, [x, y])
prob.add_sos_constraint(p2, [x, y])
tv = prob.sym_to_var(t)
prob.set_objective('min', tv)
prob.solve()
print(tv.value)
# returns t ~ 0.25