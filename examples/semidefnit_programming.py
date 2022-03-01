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

m = 15
n = 10
p = 5
np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @ P
q = np.random.randn(n)
G = np.random.randn(m, n)
h = G @ np.random.randn(n)
A = np.random.randn(p, n)
b = np.random.randn(p)

print(np.shape(h), np.shape(np.random.randn(n)))
# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h,
                  A @ x == b])
prob.solve()