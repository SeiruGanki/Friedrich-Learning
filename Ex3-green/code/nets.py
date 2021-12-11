import math
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from scipy.stats.distributions import norm

def compute_nd_sphere_area(d,R):
    return 2* np.pi **(d/2) /scipy.special.gamma(d/2) * R ** (d-1)

def compute_nd_ball_volumn(d,R):
    return np.pi ** (d/2) / scipy.special.gamma(d/2+1) * R ** d

# torch.set_default_tensor_type('torch.DoubleTensor')

def Swish2(x):
    return x**2 * torch.sigmoid(x)

degree = 1
c = 0
d = 3
Area = compute_nd_sphere_area(d,1)
Volumn = compute_nd_ball_volumn(d,1)
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
        for i in range(dim):
            Ix[i,i] = 1.
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
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1) 
            return torch.cos(np.pi/2*r)*y.squeeze(1)+ 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1) 
            return torch.cos(np.pi/2*r) ** (1) * y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
class ResNet_Tanh4(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_Tanh4, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        for i in range(dim):
            Ix[i,i] = 1.
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
                        
        y = self.outlayer(y)
        
        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*torch.exp(y).squeeze(1) + 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)

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
        for i in range(dim):
            Ix[i,i] = 1.
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
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1) 
            return torch.cos(np.pi/2*r)*torch.exp(y).squeeze(1)+ 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

class ResNet_Relu4(nn.Module):
    def __init__(self, m,dim=2,degree=1,boundary_control_type='cube'):
        super(ResNet_Relu4, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        self.outlayer = nn.Linear(m, 1, bias = False)
        self.degree = degree
        self.dim =dim

        Ix = torch.zeros([dim,m]).cuda()
        for i in range(dim):
            Ix[i,i] = 1.
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
        
        y = self.outlayer(y)

        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*torch.exp(y).squeeze(1)+ 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

class ResNet_Swish4(nn.Module):
    def __init__(self, m,dim=2,degree=1,boundary_control_type='cube'):
        super(ResNet_Swish4, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        self.outlayer = nn.Linear(m, 1, bias = False)
        self.degree = degree
        self.dim =dim

        Ix = torch.zeros([dim,m]).cuda()
        for i in range(dim):
            Ix[i,i] = 1.
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = Swish2(y)
        y = self.fc2(y)
        y = Swish2(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = Swish2(y)
        y = self.fc4(y)       
        y = Swish2(y)
        y = y+s
        
        y = self.outlayer(y)

        if self.boundary_control_type == 'none':
            return y.squeeze(1)
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*torch.exp(y).squeeze(1)+ 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

def soft_trine(x):
    out = torch.where(
        x >= 0,
        torch.abs(x - 2 * torch.floor((x + 1) / 2)),
        x / (1 + torch.abs(x))
    )
    return out

class ResNet_ST(nn.Module):  # soft-sign triangular wave
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_ST, self).__init__()
        self.fc1 = nn.Linear(dim, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        for i in range(dim):
            Ix[i,i] = 1.
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
        elif self.boundary_control_type == 'net_u':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)+ 1/(d*(d-2)*Volumn) 
        elif self.boundary_control_type == 'net_v':
            r = torch.sum(x**2,axis=1)
            return torch.cos(np.pi/4 + np.pi/4*r)*y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()




