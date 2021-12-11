# the solution is mulit-D polynomial: u(x) = (x1^2-1)*(x2^2-1)*...*(xd^2-1)
# the problem is Laplace u = f, in the domain

import torch 
from numpy import array, prod, sum, zeros, pi
import numpy as np
import pickle

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
c = 1

h = 0.0001 # step length ot compute derivative\]

# define the true solution for numpy array (N sampling points of d variables)
def true_solution(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    mask = y > 0.9*x
    z = np.zeros_like(x)
    z[mask] = np.sin(np.pi*(x[mask]+1)**2/4)*np.sin(np.pi*(y[mask]-0.9*x[mask])/2)
    z[~mask] = np.exp(-5*(x[~mask]**2+(y[~mask]-0.9*x[~mask])**2))
    return z

# define the right hand function for numpy array (N sampling points of d variables)
# x_batch has the shape of (N,d)
def f(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    mask = y > 0.9*x
    f = torch.zeros_like(x)
    f[mask] = torch.sin(np.pi*(x[mask]+1)**2/4)*torch.sin(np.pi*(y[mask]-0.9*x[mask])/2) +\
         torch.cos(np.pi*(x[mask]+1)**2/4)*torch.sin(np.pi*(y[mask]-0.9*x[mask])/2) * np.pi*(x[mask]+1) /2
    f[~mask] = torch.exp(-5*(x[~mask]**2+(y[~mask]-0.9*x[~mask])**2)) * (-10*x[~mask]) + torch.exp(-5*(x[~mask]**2+(y[~mask]-0.9*x[~mask])**2))
    return f

# specify the domain type
def domain_shape():
    return 'cube'

# output the domain parameters
def domain_parameter(d):
    intervals = zeros((d,2)) # the first interval is for times T
    for i in range(0,d):
        intervals[i,:] = array([-1,1]) # it means the radius of cylinder
    return intervals

##############

watching_data_slu = []
watching_data_phi = []

def g_d(x_batch_diri):
    x = x_batch_diri[:,0]
    y = x_batch_diri[:,1]
    mask = y > 0.9*x
    z = torch.zeros_like(x)
    z[mask] = torch.sin(np.pi*(x[mask]+1)**2/4)*torch.sin(np.pi*(y[mask]-0.9*x[mask])/2)
    z[~mask] = torch.exp(-5*(x[~mask]**2+(y[~mask]-0.9*x[~mask])**2))
    return z

def norm_vec(N,d,boundary_type):
    norm_vec_batch = torch.zeros([N,d]).cuda()
    N_perface = N//(2*d)
    for i in range(d): 
        norm_vec_batch[2*i*N_perface:(2*i+1)*N_perface,i] = -1
        norm_vec_batch[(2*i+1)*N_perface:(2*i+2)*N_perface,i] = 1
    return norm_vec_batch

def compute_phi_dirichlet_boundary_loss(x_batch_diri,phi_net):
    N_2 = list(x_batch_diri.shape)[0]
    d = list(x_batch_diri.shape)[1]  
    loss =  torch.norm(phi_net(x_batch_diri),2,0) ** 2 / N_2   
    return loss

def compute_solution_boundary_loss(u_net,x_batch_diri):
    tensor1 = torch.Tensor(true_solution(x_batch_diri))
    tensor2 = u_net(x_batch_diri)
    loss = torch.Tensor(true_solution(x_batch_diri)) - u_net(x_batch_diri)
    return torch.sum(loss **2 )/ x_batch_diri.shape[0]

def pickle_data():
    global watching_data_slu,watching_data_phi
    with open('./sequential/watching_data_slu.txt','wb') as fw:
        pickle.dump(watching_data_slu, fw)
    with open('./sequential/watching_data_phi.txt','wb') as fw:
        pickle.dump(watching_data_phi, fw)
    return 0


#### autograd module #### 
'''
we assume that the matrix A is symmetric 
''' 

def generate_matrix_A_batch(x_batch):
    N_1,d = x_batch.shape[0],x_batch.shape[1]
    A_batch = torch.ones([N_1,1,d])
    A_batch[:,0,0] = 1
    A_batch[:,0,1] = 0.9
    return A_batch
    
def compute_least_square_loss(x_batch_inside,slu_net):
    N_1,d = x_batch_inside.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True
    slu_net_output_inside = slu_net(x_batch_inside)
    grad = torch.autograd.grad(slu_net_output_inside,x_batch_inside,grad_outputs=torch.ones(slu_net_output_inside.shape),create_graph=True) 
    A_batch_inside = generate_matrix_A_batch(x_batch_inside)
    A_mul_grad = torch.bmm(A_batch_inside[:],grad[0].unsqueeze(2)).squeeze(2).squeeze(1)
    residue = torch.norm(A_mul_grad+slu_net_output_inside-f(x_batch_inside))**2/N_1
    return residue



def compute_Aphi_prod_phi_norm_autograd(x_batch_inside,x_batch_diri,slu_net,phi_net,norm_vec_batch):
    N_1,N_2,d = x_batch_inside.shape[0],x_batch_diri.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True
    phi_net_output_inside = phi_net(x_batch_inside)
    grad = torch.autograd.grad(phi_net_output_inside,x_batch_inside,grad_outputs=torch.ones(phi_net_output_inside.shape),create_graph=True)

    A_batch_inside = generate_matrix_A_batch(x_batch_inside)

    A_mul_grad = torch.bmm(A_batch_inside[:],grad[0].unsqueeze(2)).squeeze(2).squeeze(1)

    slu_net_output_inside = slu_net(x_batch_inside)

    part_1 = - torch.dot(A_mul_grad,slu_net_output_inside)/ N_1
    
    # compute tidle T norm
    A_mul_grad_norm = torch.sum((-A_mul_grad+c*phi_net_output_inside)** 2)
    phi_norm = A_mul_grad_norm / N_1 
    # end compute tidle T norm

    x_batch_diri.requires_grad = True
    A_batch_diri = generate_matrix_A_batch(x_batch_diri)
    phi_net_output_boundary = phi_net(x_batch_diri)  # 

    l = 2

    part_2_vec = torch.bmm(A_batch_diri,norm_vec_batch.unsqueeze(2)).squeeze()* phi_net_output_boundary
    part_2 = torch.dot(g_d(x_batch_diri),part_2_vec) /N_2 * (1*d /l)  # last 2 is the area/len

    part_4 = c * torch.dot(slu_net_output_inside,phi_net_output_inside) / N_1  

    part_5 = - torch.dot(f(x_batch_inside),phi_net_output_inside) / N_1 

    Aphi_prod = part_1 + part_2 + part_4 + part_5

    return Aphi_prod, phi_norm
