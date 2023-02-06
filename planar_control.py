import torch
from planar_solver import Solver
from e2c_model import E2C
from PIL import Image
import numpy as np
from datasets import *
import matplotlib.pyplot as plt
import cvxpy as cp


# load the tarned representation model
model = E2C(obs_dim=1600, z_dim=2, u_dim=2, env='planar').eval() # according to the settings in file train_e2c.py
model.load_state_dict(torch.load('result/planar/normal_train3/model_5000')) 


# set up the controller
solver = Solver(model)

# step0: random initial one image
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
dataset = datasets['planar']('./data/data/' + 'planar')
x0, u, x_next = dataset[1]
# imgplot = plt.imshow(x.squeeze(), cmap='gray')
# plt.show()
x0 = x0.view(-1, model.obs_dim).double()  # x0.shape is torch.Size([1, 1600])


# step1: take the image into the representation model to obtain the latent state
u1 = solver.mpcsolver2(x_init=x0)
print(u1)

# step2: take the latent state into the controller to get the control u

# step3: add the control u into the image to get new image (raw state)

# step4: take the new image into the step1

