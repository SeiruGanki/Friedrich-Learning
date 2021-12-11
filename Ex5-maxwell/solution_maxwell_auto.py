# the solution is mulit-D polynomial: u(x) = (x1^2-1)*(x2^2-1)*...*(xd^2-1)
# the problem is Laplace u = f, in the domain

import torch 
from numpy import array, prod, sum, zeros, pi
import numpy as np
import pickle

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
c = 0.1

h = 0.0001 # step length ot compute derivative\]

# define the true solution for numpy array (N sampling points of d variables)
def true_solution_H(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    z = x_batch[:,2]
    slu = np.zeros_like(x_batch)
    slu[:,0] = np.sin(x) * (np.cos(z)-np.cos(y))
    slu[:,1] = np.sin(y) * (np.cos(x)-np.cos(z))
    slu[:,2] = np.sin(z) * (np.cos(y)-np.cos(x))
    

    return slu

def true_solution_E(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    z = x_batch[:,2]
    slu = np.zeros_like(x_batch)
    slu[:,0] = np.sin(y)*np.sin(z)
    slu[:,1] = np.sin(z)*np.sin(x)
    slu[:,2] = np.sin(x)*np.sin(y)
    return slu

# define the right hand function for numpy array (N sampling points of d variables)
# x_batch has the shape of (N,d)
def f(x_batch):
    f = torch.zeros_like(x_batch)
    return f

def g(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    z = x_batch[:,2]
    g = torch.zeros_like(x_batch)
    g[:,0] = 3*torch.sin(y)*torch.sin(z)
    g[:,1] = 3*torch.sin(z)*torch.sin(x)
    g[:,2] = 3*torch.sin(x)*torch.sin(y)
    return g


# specify the domain type
def domain_shape():
    return 'cube'

# output the domain parameters
def domain_parameter(d):
    intervals = zeros((d,2)) # the first interval is for times T
    for i in range(0,d):
        intervals[i,:] = array([0,np.pi]) # it means the radius of cylinder
    return intervals

##############


def norm_vec(N,d,boundary_type):
    norm_vec_batch = torch.zeros([N,d]).cuda()
    N_perface = N//(2*d)
    for i in range(d): 
        norm_vec_batch[2*i*N_perface:(2*i+1)*N_perface,i] = -1
        norm_vec_batch[(2*i+1)*N_perface:(2*i+2)*N_perface,i] = 1
    return norm_vec_batch


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

#matrix_A = torch.zeros(3,3,3)
#matrix_A[0,1,2] = -1
#matrix_A[0,2,1] = 1 
#matrix_A[1,0,2] = 1
#matrix_A[1,2,0] = -1 
#matrix_A[2,0,1] = -1
#matrix_A[2,1,0] = 1 
# A_batch = torch.zeros(20000,3,3,3)
# A_batch[:] = matrix_A
#A_batch = A_batch.double().cuda()

def generate_matrix_A_batch(x_batch):
    N_1,d = x_batch.shape[0],x_batch.shape[1]
    A_batch = torch.zeros(N_1,3,3,3)
    A_batch[:] = matrix_A
    return A_batch



def compute_Aphi_prod_phi_norm_autograd(x_batch_inside,mu,sigma,H_net,E_net,phiH_net,phiE_net,A_batch):
    N_1,d = x_batch_inside.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True

    phiH_net_output_inside = phiH_net(x_batch_inside)
    phi_norm = 0
    part_1 = 0

    Jacobian_phiH = torch.zeros(N_1,3,3,1)
    phiE_net_output_inside = phiE_net(x_batch_inside)
    for i in range(d):
        Jacobian_phiH[:,:,i,:] = torch.autograd.grad(phiH_net_output_inside[:,i],\
            x_batch_inside,grad_outputs=torch.ones(phiH_net_output_inside[:,i].shape),create_graph=True)[0].unsqueeze(2)
  
    # A_batch_inside = generate_matrix_A_batch(x_batch_inside)
    A_batch_inside = A_batch

    A_mul_grad = torch.sum(torch.bmm(A_batch_inside.view(N_1 * 3,3,3),Jacobian_phiH.view(N_1 * 3,3,1)).view(N_1,3,3,1).squeeze(3),dim=1)
    
    E_net_output_inside = E_net(x_batch_inside)

    part_11 = torch.dot(A_mul_grad.view(-1),E_net_output_inside.view(-1))/ N_1

    A_mul_grad_norm = torch.sum((A_mul_grad+ sigma * phiE_net_output_inside)** 2)
    phi_norm += A_mul_grad_norm / N_1 


    Jacobian_phiE = torch.zeros(N_1,3,3,1)
    for i in range(d):
        Jacobian_phiE[:,:,i,:] = torch.autograd.grad(phiE_net_output_inside[:,i],\
            x_batch_inside,grad_outputs=torch.ones(phiE_net_output_inside[:,i].shape),create_graph=True)[0].unsqueeze(2)
   
    A_mul_grad = torch.sum(torch.bmm(A_batch_inside.view(N_1 * 3,3,3),Jacobian_phiE.view(N_1 * 3,3,1)).view(N_1,3,3,1).squeeze(3),dim=1)

    H_net_output_inside = H_net(x_batch_inside)

    part_12 = - torch.dot(A_mul_grad.view(-1),H_net_output_inside.view(-1))/ N_1
    #part_1 = part_12+part_11

    A_mul_grad_norm = torch.sum((-A_mul_grad+mu*phiH_net_output_inside)** 2) 
    phi_norm += A_mul_grad_norm / N_1 

 
    part_41 = mu * torch.dot(H_net_output_inside.view(-1),phiH_net_output_inside.view(-1)) / N_1  
    part_42 = sigma * torch.dot(E_net_output_inside.view(-1),phiE_net_output_inside.view(-1)) / N_1 
    #part_4 = part_41+part_42
    
    part_51 = - torch.dot(f(x_batch_inside).view(-1),phiH_net_output_inside.view(-1)) / N_1 
    part_52 = - torch.dot(g(x_batch_inside).view(-1),phiE_net_output_inside.view(-1)) / N_1 
    #part_5 = part_51+part_52
    
    Aphi_prod1 = part_12  + part_41 + part_51
    Aphi_prod2 = part_11  + part_42 + part_52

    return Aphi_prod1,Aphi_prod2, phi_norm

