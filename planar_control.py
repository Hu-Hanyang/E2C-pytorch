import torch
from planar_solver import Solver
from e2c_model import E2C
from PIL import Image
import numpy as np
from datasets import *
import matplotlib.pyplot as plt


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
# mu, logvar = model.encoder(x0)
# z_bar_t = model.reparam(mu, logvar).detach()
# print(z_bar_t.shape)

# h_t = model.net(z_bar_t) 
# B_t = model.fc_B(h_t)
# o_t = model.fc_o(h_t)

# v_t, r_t = model.fc_A(h_t).chunk(2, dim=1)
# v_t = torch.unsqueeze(v_t, dim=-1)
# r_t = torch.unsqueeze(r_t, dim=-2)

# A_t = torch.eye(model.z_dim).repeat(z_bar_t.size(0), 1, 1) + torch.bmm(v_t, r_t)
# print(f'The shape of A matrix is {A_t.shape}')
# B_t = B_t.view(-1, model.z_dim, model.u_dim)
# print(f'The shape of B matrix is {B_t.shape}')
# u_t = torch.tensor([1.0, 1.0]).unsqueeze(0)
# print(f'The shape of the u_t is {u_t.shape}')
# print(f'The shape of u_t.unsqueeze(-1) is {u_t.unsqueeze(-1).shape}')
# print(f'The shape of z_bar_t.unsqueeze(-1) is {z_bar_t.unsqueeze(-1).shape}')
# z_next = A_t.bmm(z_bar_t.unsqueeze(-1)).squeeze(-1) + B_t.bmm(u_t.unsqueeze(-1)).squeeze(-1) + o_t
# print(z_next)





# step1: take the image into the representation model to obtain the latent state
u1 = solver.mpcsolver2(x_init=x0)
print(u1)

# step2: take the latent state into the controller to get the control u

# step3: add the control u into the image to get new image (raw state)

# step4: take the new image into the step1

