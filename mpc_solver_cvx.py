import cvxpy as cp
import numpy as np

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
        self.trans = model.dynamics  # predict the next latent state

    def mpcsolver(self, x_init, predict_steps):
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

