import cvxpy as cp
import numpy as np
from normal import *


class Solver:
    """
    Solvers using E2C model with Casadi.
    """
    def __init__(self, model) -> None:
        """
        Initilize with the E2C model
        Parameters:
            model: the loaded E2C model
        """
        self.z_dim = model.z_dim  # latent state dimension
        self.u_dim = model.u_dim  # control dimensions
        self.trans = model.transition # here needs to be revised!!!
        self.reparam = model.reparam
        self.encoder = model.encoder
        self.T = 10  # prediction horizon

    def mpcsolver(self, x_init):
        """
        MPC solver based on the E2C model
        Parameters:
            x_init: tensor, shape:[]
                raw pictures.
            predict_steps: int
                steps for predict horizon.
        Returns:
            state: np.array, shape: [predict_steps+1, state_dimension]
            controls: np.array, shape: [predict_steps, control_dimension]
        """
        Rz = np.eye(self.z_dim) * 0.1
        Ru = np.eye(self.u_dim)
        z = cp.Variable((self.z_dim, self.T+1)) # latent representation
        u = cp.Variable((self.u_dim, self.T))
        cost = 0
        constr = []
        mu, logvar = self.encoder(x_init)
        z0 = self.reparam(mu, logvar).numpy()
        for t in range(self.T):
            cost += cp.quad_form(z[:, t+1], Rz) + cp.quad_form(u[:, t], Ru)
            constr += [z[:, t+1] == self.trans(z[:, t], u[:, t])] # here maybe lies the problem
        constr += [z[:, 0] == z0]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()




