# this file is used to test anything used in the reimplementation unformally.
from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain
from ilqr.examples.cartpole import CartpoleDynamics


dt = 0.05
pole_length = 1.0
dynamics = CartpoleDynamics(dt, l=pole_length)