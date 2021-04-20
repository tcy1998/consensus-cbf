# import cvxpy as cp
# import numpy as np

# # Generate a random SDP.
# n = 3
# p = 3
# np.random.seed(1)
# C = np.random.randn(n, n)
# A = []
# b = []
# for i in range(p):
#     A.append(np.random.randn(n, n))
#     b.append(np.random.randn())

# # Define and solve the CVXPY problem.
# # Create a symmetric matrix variable.
# X = cp.Variable((n,n), symmetric=True)
# # The operator >> denotes matrix inequality.
# constraints = [X >> 0]
# constraints += [
#     cp.trace(A[i] @ X) == b[i] for i in range(p)
# ]
# prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
#                   constraints)
# prob.solve()

# # Print result.
# print("The optimal value is", prob.value)
# print("A solution X is")
# print(X.value)


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