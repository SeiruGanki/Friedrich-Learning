from matplotlib.pyplot import xticks
from utils import *
from settings import * 

import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

d = 2
domain_intervals = domain_parameter(2)
torch.manual_seed(0)
np.random.seed(0)
N_inside_train =45000
N_boundary_train = 5000

x1_train = data_to_cuda(generate_uniform_points_in_fan(domain_intervals,N_inside_train,0,np.pi))
print(x1_train)

x2_train = data_to_cuda(generate_uniform_points_on_fan(domain_intervals,N_boundary_train,0,np.pi,direction='outflow')) # multiply by 2 since we will discard half of them
print(x2_train)

norm_vec_batch = norm_vec(N_boundary_train,d)  # multiply by 2 since we will discard half of them


def true_solution2net(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    r2 = x**2 + y**2
    mask = r2 > (1/2)**2
    z = torch.zeros_like(x)
    z[mask] = 1
    z[~mask] = 0
    return z

def test_function2net(x_batch):
    return (-np.pi/2-torch.atan(-x_batch[:,0]/x_batch[:,1]))

from torchsummary import summary
net_u = generate_network('ResNet_Relu',d,150,net_type = 'net_u')
summary(net_u, (2,))
# print(compute_Friedrich_loss(true_solution2net,test_function2net,x1_train,x2_train,norm_vec_batch))
# print(compute_PINN_loss(true_solution2net,x1_train))