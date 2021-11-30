from cvxopt import matrix

import cvxpy as cp
import numpy as np
import cvxopt
import time




x = cp.Variable((3, 3), PSD=True)
y = cp.Variable((1, 3))
a = cvxopt.matrix([[1, 0, 0], [0, 0, 0], [0, 1, 1]])
b = cvxopt.matrix([[1,-1,0]])
print(b.T@x@b)
constraints = [b.T@x@b+y@b >> 0]
objective = cp.Minimize(cp.trace(x-np.eye(3))+cp.norm(y,1))
start_time = time.time()
prob = cp.Problem(objective, constraints)
prob.solve()

print("The optimal value is", prob.value)
print("A solution X is")
print(x.value)
print("A solution Y is")
print(y.value)
print("--- %s seconds ---" % (time.time() - start_time))