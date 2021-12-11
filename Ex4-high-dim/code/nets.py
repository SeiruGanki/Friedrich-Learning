import math
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.set_default_tensor_type('torch.DoubleTensor')

degree = 1
c = 0

"""
define neutral network 
"""

def g(x,if_cuda=False):
    if if_cuda == False:
        value = np.zeros(x.shape[0])
        mask = x > 0
        value[mask] = 1
    else:
        value = torch.zeros(x.shape[0])
        mask = x > 0
        value[mask] = 1
    return value

def Swish(x):
    return x * torch.sigmoid(x)
class ResNet_Swish(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_Swish, self).__init__()
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
        y = Swish(y)
        y = self.fc2(y)
        y = Swish(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = Swish(y)
        y = self.fc4(y)       
        y = Swish(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = Swish(y)
        y = self.fc6(y)    
        y = Swish(y)
        y = y+s
                
        y = self.outlayer(y)

        if self.boundary_control_type == 'none':
            return y.squeeze(1)

        elif self.boundary_control_type == 'net_u':
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

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
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 

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
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 

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
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

class ResNet_Relu4(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ResNet_Relu4, self).__init__()
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
        self.degree = degree
        self.outlayer = nn.Linear(m, 1, bias = False)

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
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 
            
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
            sum_w_2 = torch.sum(x[:,2:],axis=1) **2
            base =  g(torch.exp(2*x[:,0])-4*x[:,1] *(1+torch.exp(-sum_w_2)),if_cuda=True)

            return  torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]) *y.squeeze(1) + base *  (1-torch.sin(np.pi/2*x[:,0])* torch.sin(np.pi/2*x[:,1]))

        elif self.boundary_control_type == 'net_v':
            return  torch.cos(np.pi/2*x[:,0])* torch.cos(np.pi/2*x[:,1]) * y.squeeze(1) 

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()




