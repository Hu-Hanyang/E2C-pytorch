import torch
from planar_solver import Solver
from e2c_model import E2C

# load the tarned representation model
model = E2C(obs_dim=1600, z_dim=2, u_dim=2, env='planar') # according to the settings in file train_e2c.py
model.load_state_dict(torch.load('result/planar/normal_train2/model_5000'))

# set up the controller
solver = Solver(model)

# step0: random initial one image

# step1: take the image into the representation model to obtain the latent state

# step2: take the latent state into the controller to get the control u

# step3: add the control u into the image to get new image (raw state)

# step4: take the new image into the step1

