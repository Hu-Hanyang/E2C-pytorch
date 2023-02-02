from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys

from normal import *
from e2c_model import E2C
from datasets import *
import data.sample_planar as planar_sampler
import data.sample_pendulum_data as pendulum_sampler
import data.sample_cartpole_data as cartpole_sampler



device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': GymPendulumDatasetV2}
settings = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1)}
samplers = {'planar': planar_sampler, 'pendulum': pendulum_sampler, 'cartpole': cartpole_sampler}
num_eval = 10 # number of images evaluated on tensorboard

dataset = datasets['planar']('./data/data/' + 'planar')
x, u, x_next = dataset[0]
imgplot = plt.imshow(x.squeeze(), cmap='gray')
plt.show()
print (np.array(u, dtype=float))
imgplot = plt.imshow(x_next.squeeze(), cmap='gray')
plt.show()