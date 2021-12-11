import torch 
import numpy as np
from utils import * 
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')

c = 0 # the coeff of 0 order term

# define the true solution for numpy array (N sampling points of d variables)
def true_solution(x_batch):
    d = x_batch.shape[1]
    Area = compute_nd_sphere_area(d,1)
    Volumn = compute_nd_ball_volumn(d,1)
    if d == 2:
        x_abs = (x_batch[:,0]**2 + x_batch[:,1]**2) ** (1/2)
        return -1/(2*np.pi) * np.log(x_abs)
    else:
        x_abs = np.sum(x_batch**2,axis=1) ** (1/2)
        return  1/(d*(d-2)*Volumn) * (1/x_abs**(d-2))

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
    Volumn = compute_nd_ball_volumn(d,1)
    return 1/(d*(d-2)*Volumn) * torch.ones(x_batch_diri.shape[0])

def norm_vec(N,d,x_batch_diri): # return the normal vector of the boundary points 
    return x_batch_diri

def generate_matrix_A_batch(x_batch):
    N_1,d = x_batch.shape[0],x_batch.shape[1]
    batch = torch.ones([N_1,d,d])
    return -batch * torch.eye(d)
    
def compute_Aphi_prod_phi_norm_autograd(x_batch_inside,x_batch_diri,slu_net,phi_net,norm_vec_batch):
    N_1,N_2,d = x_batch_inside.shape[0],x_batch_diri.shape[0],x_batch_inside.shape[1]
    Area = compute_nd_sphere_area(d,1)
    Volumn = compute_nd_ball_volumn(d,1)

    x_batch_inside.requires_grad = True
    phi_net_output_batch = phi_net(x_batch_inside)
    grad = torch.autograd.grad(phi_net_output_batch,x_batch_inside,grad_outputs=torch.ones(phi_net_output_batch.shape),create_graph=True)

    A_batch_inside = generate_matrix_A_batch(x_batch_inside)

    A_mul_grad = torch.bmm(A_batch_inside,grad[0].unsqueeze(2)).squeeze(2)

    div_A_mul_grad = torch.zeros(N_1)

    for idx in range(d):  
        div_A_mul_grad += torch.autograd.grad(A_mul_grad[:,idx],x_batch_inside,grad_outputs=torch.ones_like(A_mul_grad[:,idx]), create_graph=True)[0][:,idx]

    slu_net_output_batch = slu_net(x_batch_inside)
    part_1 = - torch.dot(div_A_mul_grad,slu_net_output_batch)/ N_1 * Volumn

    A_mul_grad_norm = torch.sum(div_A_mul_grad** 2)

    phi_norm = A_mul_grad_norm / N_1 * Volumn**2

    x_batch_diri.requires_grad = True
    A_batch_diri = generate_matrix_A_batch(x_batch_diri)
  
    phi_net_output_batch = phi_net(x_batch_diri)  
    
    grad_2 = torch.autograd.grad(phi_net_output_batch,x_batch_diri,grad_outputs=torch.ones(phi_net_output_batch.shape),create_graph=True)

    part_2_vec = torch.bmm(torch.bmm(norm_vec_batch.unsqueeze(1), A_batch_diri),grad_2[0].unsqueeze(2)).squeeze()
    part_2 = torch.dot(g_d(x_batch_diri),part_2_vec) /N_2 * Area 

    part_5 = phi_net(torch.zeros([1,d])).squeeze()

    Aphi_prod = part_1 + part_2+ part_5

    return Aphi_prod, phi_norm

def compute_Friedrich_loss(net_u,net_v,x1_train,x2_train,norm_vec_batch):
    Aphi_prod, phi_norm = compute_Aphi_prod_phi_norm_autograd(x1_train,x2_train,net_u,net_v,norm_vec_batch) 
    return Aphi_prod**2/phi_norm


def compute_PINN_loss(slu_net,x_batch_inside):
    N_1,d = x_batch_inside.shape[0],x_batch_inside.shape[1]
    x_batch_inside.requires_grad = True
    slu_net_output_inside = slu_net(x_batch_inside)
    grad = torch.autograd.grad(slu_net_output_inside,x_batch_inside,grad_outputs=torch.ones(slu_net_output_inside.shape),create_graph=True) 
    A_batch_inside = generate_matrix_A_batch(x_batch_inside)
    A_mul_grad = torch.bmm(A_batch_inside[:],grad[0].unsqueeze(2)).squeeze(2)
    div_A_mul_grad = torch.zeros(N_1)

    for idx in range(d):  
        div_A_mul_grad += torch.autograd.grad(A_mul_grad[:,idx],x_batch_inside,grad_outputs=torch.ones_like(A_mul_grad[:,idx]), create_graph=True)[0][:,idx]

    loss = torch.sum((div_A_mul_grad+c*slu_net_output_inside-f(x_batch_inside))**2)/N_1
    return loss

def compute_boundary_loss(slu_net,x_batch_diri):
    N_2 = x_batch_diri.shape[0]
    return torch.sum((slu_net(x_batch_diri) - g_d(x_batch_diri)) ** 2)/N_2
