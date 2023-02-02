from torch import nn

from normal import *
from networks import *
from e2c_model import*

model = E2C(obs_dim=1600, z_dim=2, u_dim=2)
