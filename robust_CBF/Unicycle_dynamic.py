import numpy as np
import torch

class Unicycle_dynamic:
    def __init__(self) -> None:
        self.t = 0.02   # The frequency set to 50hz
        self.d = 3      # The imension of state
        self.m = 2      # The dimension of control input
        self.K = 2500   # The number of sample

        self.mu = 0     # The mean of the noise 
        self.sigma = 1  # The variance of the noise

    def dynamic(self, X, U):
        dW = torch.Tensor(np.random.normal(self.mu, self.sigma,\
            size=(self.d, self.K))) # The noise of the dynamic size is (d,K)
        x_d = torch.cos(X[2]) * U[0]
        y_d = torch.sin(X[2]) * U[0]
        w_d = U[1]
        X_d = torch.vstack((x_d, y_d, w_d))
        X_new = X_d * self.t + dW

        return X_new

    def cost_f(self, X, U):
        

if __name__ == "__main__":

    mdl = Unicycle_dynamic()
    x = torch.Tensor(np.zeros((mdl.d, mdl.K)))
    u = torch.Tensor(np.zeros((mdl.m, mdl.K)))
    State = []
    for t in range(500):
        x_next = mdl.dynamic(x, u)
        State.append(x_next)

    # print(State.size())