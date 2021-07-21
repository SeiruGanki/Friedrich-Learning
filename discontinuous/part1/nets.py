import math
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.set_default_tensor_type('torch.DoubleTensor')

degree = 1
c = 1
"""
define neutral network 
"""
class ResNet_Tanh(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_Tanh, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        self.Ix = Ix
        self.dim =dim
        self.outlayer = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = torch.tanh(y)
        y = self.fc2(y)
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = torch.tanh(y)
        y = self.fc4(y)       
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = torch.tanh(y)
        y = self.fc6(y)    
        y = torch.tanh(y)
        y = y+s
                
        y = self.outlayer(y)
        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'u_net':
            x_1 = x[:,0]
            x_2 = x[:,1]
            mask = x_2 > (0.5*x_1-0.4)
            base = torch.zeros_like(x_1)
            base[~mask] = torch.exp(-5*(x_1[~mask]**2+(-1-0.9*x_1[~mask])**2)) +\
                              torch.exp(-5*(1+(x_2[~mask]+0.9)**2)) -\
                              np.exp(-5*(1+(-1+0.9)**2)) 
            return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + base

        elif self.boundary_control_type == 'v_net':
            return  torch.cos(np.pi/4+np.pi/4*x[:,0])*torch.cos(np.pi/4+np.pi/4*x[:,1])*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    

    
class ResNet_Relu(nn.Module):
    def __init__(self, m,dim=2,degree=1,boundary_control_type='cube'):
        super(ResNet_Relu, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        self.outlayer = nn.Linear(m, 1, bias = False)
        self.degree = degree
        self.dim =dim

        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = F.relu(y**self.degree)
        y = self.fc2(y)
        y = F.relu(y**self.degree)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = F.relu(y**self.degree)
        y = self.fc4(y)       
        y = F.relu(y**self.degree)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = F.relu(y**self.degree)
        y = self.fc6(y)    
        y = F.relu(y**self.degree)
        y = y+s

        y = self.outlayer(y)

        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'u_net':
            x_1 = x[:,0]
            x_2 = x[:,1]
            mask = x_2 > (0.5*x_1-0.4)
            base = torch.zeros_like(x_1)
            base[~mask] = torch.exp(-5*(x_1[~mask]**2+(-1-0.9*x_1[~mask])**2)) +\
                              torch.exp(-5*(1+(x_2[~mask]+0.9)**2)) -\
                              np.exp(-5*(1+(-1+0.9)**2))
            return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + base    

        elif self.boundary_control_type == 'v_net':
            return  torch.cos(np.pi/4+np.pi/4*x[:,0])*torch.cos(np.pi/4+np.pi/4*x[:,1])*y.squeeze(1)


    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

def soft_trine(input):
    mask = input >= 0
    output = torch.zeros_like(input)
    output[mask] = torch.abs(input[mask] - 2 * torch.floor((input[mask]+1)/2))
    output[~mask] = input[~mask] / (torch.abs(input[~mask]) + 1)
    return output


class ResNet_ST(nn.Module):  # soft-sign triangular wave
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_Tanh, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        self.Ix = Ix
        self.dim =dim
        self.outlayer = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = soft_trine(y)
        y = self.fc2(y)
        y = soft_trine(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = soft_trine(y)
        y = self.fc4(y)       
        y = soft_trine(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = soft_trine(y)
        y = self.fc6(y)    
        y = soft_trine(y)
        y = y+s
                
        y = self.outlayer(y)
        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'u_net':
            x_1 = x[:,0]
            x_2 = x[:,1]
            mask = x_2 > (0.5*x_1-0.4)
            base = torch.zeros_like(x_1)
            base[~mask] = torch.exp(-5*(x_1[~mask]**2+(-1-0.9*x_1[~mask])**2)) +\
                              torch.exp(-5*(1+(x_2[~mask]+0.9)**2)) -\
                              np.exp(-5*(1+(-1+0.9)**2)) 
            return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + base

        elif self.boundary_control_type == 'v_net':
            return  torch.cos(np.pi/4+np.pi/4*x[:,0])*torch.cos(np.pi/4+np.pi/4*x[:,1])*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()




#----------------------- extension ---------------------------

class ReLUResNet_base(nn.Module):
    def __init__(self, m,dim=2,degree=1,boundary_control_type='cube',base_function=None):
        super(ReLUResNet_base, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        self.base_function = base_function
        self.outlayer = nn.Linear(m, 1, bias = False)
        self.degree = degree
        self.dim =dim

        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = F.relu(y**self.degree)
        y = self.fc2(y)
        y = F.relu(y**self.degree)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = F.relu(y**self.degree)
        y = self.fc4(y)       
        y = F.relu(y**self.degree)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = F.relu(y**self.degree)
        y = self.fc6(y)    
        y = F.relu(y**self.degree)
        y = y+s

        y = self.outlayer(y)
        if self.base_function == None:
            if self.boundary_control_type == 'none':
                return y.squeeze(1)
            elif self.boundary_control_type == 'u_net':
                x_1 = x[:,0]
                x_2 = x[:,1]
                mask = x_2 > (-0.4 + 0.5*x_1)
                base = torch.zeros_like(x_1)
                base[~mask] = torch.exp(-5*(x_1[~mask]**2+(-1-0.9*x_1[~mask])**2)) +\
                              torch.exp(-5*(1+(x_2[~mask]+0.9)**2)) -\
                              np.exp(-5*(1+(-1+0.9)**2)) 
                return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + base

            elif self.boundary_control_type == 'v_net':
                return  torch.cos(np.pi/4+np.pi/4*x[:,0])*torch.cos(np.pi/4+np.pi/4*x[:,1])*y.squeeze(1)
        else:
            if self.boundary_control_type == 'u_net':
                x_1 = x[:,0]
                x_2 = x[:,1]            
                return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + self.base_function(x)


    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

class TanhResNet_base(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube',base_function=None):
        super(TanhResNet_base, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        self.Ix = Ix
        self.dim =dim
        self.base_function = base_function
        self.outlayer = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = torch.tanh(y)
        y = self.fc2(y)
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = torch.tanh(y)
        y = self.fc4(y)       
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = torch.tanh(y)
        y = self.fc6(y)    
        y = torch.tanh(y)
        y = y+s
                
        y = self.outlayer(y)
        if self.base_function == None:
            if self.boundary_control_type == 'none':
                return y.squeeze(1)
            elif self.boundary_control_type == 'u_net':
                x_1 = x[:,0]
                x_2 = x[:,1]
                mask = x_2 > (-0.4 + 0.5*x_1)
                base = torch.zeros_like(x_1)
                base[~mask] = torch.exp(-5*(x_1[~mask]**2+(-1-0.9*x_1[~mask])**2)) +\
                              torch.exp(-5*(1+(x_2[~mask]+0.9)**2)) -\
                              np.exp(-5*(1+(-1+0.9)**2))
                              
            
                return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + base

            elif self.boundary_control_type == 'v_net':
                return  torch.cos(np.pi/4+np.pi/4*x[:,0])*torch.cos(np.pi/4+np.pi/4*x[:,1])*y.squeeze(1)
        else:
            if self.boundary_control_type == 'u_net':
                x_1 = x[:,0]
                x_2 = x[:,1]            
                return torch.cos(-np.pi/4+np.pi/4*x[:,0])*torch.cos(-np.pi/4+np.pi/4*x[:,1])*y.squeeze(1) + self.base_function(x)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
