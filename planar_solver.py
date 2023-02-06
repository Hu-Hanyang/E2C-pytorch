import cvxpy as cp
import numpy as np
from normal import *
from ilqr import *


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
        self.encoder = model.encoder
        self.reparam = model.reparam
        self.trans = model.dynamics 
        self.T = 10  # prediction horizon
        self.state_trans = model.state_trans

    def mpcsolver1(self, x_init): # not finished
        """
        MPC solver1 based on the E2C model
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
        # preprocess the real input raw image x_t
        mu, logvar = self.encoder(x_init)
        z0 = self.reparam(mu, logvar).detach().numpy()
        for t in range(self.T):
            cost += cp.quad_form(z[:, t+1], Rz) + cp.quad_form(u[:, t], Ru)
            constr += [z[:, t+1] == self.trans(z[:, t], u[:, t])] # here maybe lies the problem
        constr += [z[:, 0] == z0]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        return u.value[0]
    
    def mpcsolver2(self, x_init, x_goal):
        """
        MPC solver2 based on the E2C model
        Parameters:
            x_init: tensor, 
                raw pictures.
            x_goal: tensor, 
                goal state raw pictures.
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
        # preprocess the real input raw image x_t
        mu, logvar = self.encoder(x_init)
        z0 = self.reparam(mu, logvar).detach()
        for t in range(self.T):
            cost += cp.quad_form(z[:, t+1], Rz) + cp.quad_form(u[:, t], Ru)
            constr += [z[:, t+1] == self.state_trans(z[:, t], u[:, t])] # here the type of z could be a problem
        constr += [z[:, 0] == z0]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        return u.value[0]

    def lqrsolver(self):
        pass
    




