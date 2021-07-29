from matplotlib.pyplot import xticks
from utils import *
from settings import * 

import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

d = 2
domain_intervals = domain_parameter(2)
N_test = 8
N_boundary_train = 8

x1_train = data_to_cuda(generate_uniform_points_in_cube(domain_intervals,N_test,generate_type='lhs'))

x2_train = data_to_cuda(generate_uniform_points_on_cube(domain_intervals,2*N_boundary_train//4,generate_type='lhs')) # multiply by 2 since we will discard half of them
x2_train = take_inflow_bd(x2_train,if_cuda=True)

norm_vec_batch = norm_vec(2*N_boundary_train,d)  # multiply by 2 since we will discard half of them
norm_vec_batch = take_inflow_bd(norm_vec_batch,if_cuda=True)

print(x2_train)
def true_solution2net(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    mask = x > y
    z = torch.zeros_like(x)
    z[mask] = 1
    return z

def test_function2net(x_batch):
    return torch.cos(x_batch[:,0] * np.pi/2) * torch.cos(x_batch[:,1] * np.pi/2)

print(compute_Friedrich_loss(true_solution2net,test_function2net,x1_train,x2_train,norm_vec_batch))