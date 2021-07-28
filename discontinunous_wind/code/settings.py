# the solution is mulit-D polynomial: u(x) = (x1^2-1)*(x2^2-1)*...*(xd^2-1)
# the problem is Laplace u = f, in the domain

import torch 
import numpy as np

# torch.set_default_tensor_type('torch.cuda.DoubleTensor')

c = 0 # the coeff of 0 order term

# define the true solution for numpy array (N sampling points of d variables)
def true_solution(x_batch):
    x = x_batch[:,0]
    y = x_batch[:,1]
    r2 = x**2 + y**2
    mask = r2 > (1/2)**2
    z = np.zeros_like(x)
    z[mask] = 1
    z[~mask] = 0
    return z

# define the right hand function for numpy array (N sampling points of d variables)
# x_batch has the shape of (N,d)
def f(x_batch):
    return torch.zeros(x_batch.shape[0])

# output the domain parameters
def domain_parameter(d):
    intervals = np.zeros((d,2)) # the first interval is for times T
    for i in range(0,d):
        intervals[i,:] = np.array([0,1]) # it means the radius of cylinder
    return intervals

def g_d(x_batch_diri): # the boundary function
    x = x_batch_diri[:,0]
    mask = x > 1/2
    z = torch.zeros_like(x)
    z[mask] = 1
    return z

def norm_vec(N,d): # return the normal vector of the boundary points 
    norm_vec_batch = torch.zeros([N,d]).cuda()
    norm_vec_batch[:,1] = -1
    return norm_vec_batch


def generate_matrix_A_batch(x_batch):
    N_1,d = x_batch.shape[0],x_batch.shape[1]
    x = x_batch[:,0]
    y = x_batch[:,1]
    A_batch = torch.ones([N_1,1,d])
    A_batch[:,0,0] = -y[:]
    A_batch[:,0,1] = x[:]
    return A_batch
    

def compute_Aphi_prod_phi_norm_autograd(x_batch_inside,x_batch_diri,slu_net,phi_net,norm_vec_batch):
    N_1,N_2,d = x_batch_inside.shape[0],x_batch_diri.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True
    phi_net_output_inside = phi_net(x_batch_inside)
    grad = torch.autograd.grad(phi_net_output_inside,x_batch_inside,grad_outputs=torch.ones(phi_net_output_inside.shape),create_graph=True)

    A_batch_inside = generate_matrix_A_batch(x_batch_inside)

    A_mul_grad = torch.bmm(A_batch_inside[:],grad[0].unsqueeze(2)).squeeze(2).squeeze(1)

    slu_net_output_inside = slu_net(x_batch_inside)

    part_1 = - torch.dot(A_mul_grad,slu_net_output_inside)/ N_1 * np.pi/4
    
    # compute tidle T norm
    A_mul_grad_norm = torch.sum((-A_mul_grad+c*phi_net_output_inside)** 2) * np.pi/4 ** 2
    phi_norm = A_mul_grad_norm / N_1 
    # end compute tidle T norm

    x_batch_diri.requires_grad = True
    A_batch_diri = generate_matrix_A_batch(x_batch_diri)
    phi_net_output_boundary = phi_net(x_batch_diri)  # 

    l = 2

    part_2_vec = torch.bmm(A_batch_diri,norm_vec_batch.unsqueeze(2)).squeeze()* phi_net_output_boundary
    part_2 = torch.dot(g_d(x_batch_diri),part_2_vec) /N_2 * 1  # last 2 is the area/len

    part_4 = c * torch.dot(slu_net_output_inside,phi_net_output_inside) / N_1 * np.pi/4

    part_5 = - torch.dot(f(x_batch_inside),phi_net_output_inside) / N_1 * np.pi/4

    Aphi_prod = part_1 + part_2 + part_4 + part_5

    return Aphi_prod, phi_norm

def compute_Friedrich_loss(net_u,net_v,x1_train,x2_train,norm_vec_batch):
    Aphi_prod, phi_norm = compute_Aphi_prod_phi_norm_autograd(x1_train,x2_train,net_u,net_v,norm_vec_batch)  # apply algorithms
    return Aphi_prod**2/phi_norm


def compute_PINN_loss(slu_net,x_batch_inside):
    N_1,d = x_batch_inside.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True
    slu_net_output_inside = slu_net(x_batch_inside)
    grad = torch.autograd.grad(slu_net_output_inside,x_batch_inside,grad_outputs=torch.ones(slu_net_output_inside.shape),create_graph=True) 
    A_batch_inside = generate_matrix_A_batch(x_batch_inside)
    A_mul_grad = torch.bmm(A_batch_inside[:],grad[0].unsqueeze(2)).squeeze(2).squeeze(1)
    loss = torch.norm(A_mul_grad+c*slu_net_output_inside-f(x_batch_inside))**2/N_1
    return loss

def compute_boundary_loss(slu_net,x_batch_diri):
    N_2 = x_batch_diri.shape[0]
    return torch.sum((slu_net(x_batch_diri) - g_d(x_batch_diri)) ** 2 )/N_2
