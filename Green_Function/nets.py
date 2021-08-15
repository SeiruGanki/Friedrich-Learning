import math
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.set_default_tensor_type('torch.DoubleTensor')

degree = 1
c = 0
d = 10
frac = 1/5
term = 2 ** d

"""
define neutral network 
"""
def g(x,if_cuda=False):
    if if_cuda == False:
        value = np.zeros(x.shape[0])
        mask = x > 0
        value[mask] = np.exp(-x[mask])
        value[~mask] = -np.cos(x[~mask])
    else:
        value = torch.zeros(x.shape[0])
        mask = x > 0
        value[mask] =  torch.exp(-x[mask])
        value[~mask] = -torch.cos(x[~mask])
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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            # return  torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) **(1/10) * y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            return  term * torch.prod(x[:,:],dim=1)**(frac) * y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            #return  term * torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac) * y.squeeze(1) + base *(1-term*torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))
            # return  torch.min(x[:,:],1)[0]*y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            #return  y.squeeze(1) 
        elif self.boundary_control_type == 'net_v':

            return  torch.prod(torch.cos(np.pi/4+np.pi/4*x[:,:]),dim=1) *y.squeeze(1)
            # return torch.min(x[:,:],1)[0]*y.squeeze(1)

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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            return  torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) **(1/10) * y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            # return  torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) * y.squeeze(1) + base *(1-term*torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))
            # return  torch.min(x[:,:],1)[0]*y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])

        elif self.boundary_control_type == 'net_v':
            return  torch.prod(torch.cos(np.pi/4+np.pi/4*x[:,:]),dim=1) *y.squeeze(1) # *y.squeeze(1) 
            # return  torch.prod(1-x[:,:],dim=1) *y.squeeze(1)
            # return torch.min(x[:,:],1)[0]*y.squeeze(1)

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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            return torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) *y.squeeze(1) + base *(1-term*torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))

        elif self.boundary_control_type == 'net_v':
            return  torch.prod(torch.cos(np.pi/2*x[:,:]),dim=1) *y.squeeze(1)

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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            # return  torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) **(1/10) * y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            return  torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac) * y.squeeze(1) + base *(1-torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))
            # return  torch.min(x[:,:],1)[0]*y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            # return  term * torch.prod(x[:,:],dim=1)**(frac) * y.squeeze(1) + base *  (1-torch.min(x[:,:],1)[0])
            # return  y.squeeze(1) 
        elif self.boundary_control_type == 'net_v':

            return  torch.prod(torch.cos(np.pi/4+np.pi/4*x[:,:]),dim=1)*y.squeeze(1)
            # return torch.min(x[:,:],1)[0]*y.squeeze(1)


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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            return torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) *y.squeeze(1) + base *(1-term*torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))

        elif self.boundary_control_type == 'net_v':
            return  torch.prod(torch.cos(np.pi/2*x[:,:]),dim=1) *y.squeeze(1)
            
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
            base =  g(torch.exp(x[:,0])-2/(d-1) * torch.sum(x[:,1:],axis=1)*np.exp(1/2),if_cuda=True)
            return torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1) *y.squeeze(1) + base *(1-term*torch.prod(torch.sin(np.pi/4*x[:,:]),dim=1)**(frac))

        elif self.boundary_control_type == 'net_v':
            return  torch.prod(torch.cos(np.pi/2*x[:,:]),dim=1) *y.squeeze(1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()




#----------------------- extension ---------------------------

class ResNet_Relu_base(nn.Module):
    def __init__(self, m,dim=2,degree=1,boundary_control_type='cube',base_function=None):
        super(ResNet_Relu_base, self).__init__()
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
            elif self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2
                mask = x_1 > -1/2
                base = torch.ones_like(x_1)
                base[mask] = 0
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r)*y.squeeze(1)+base

            elif self.boundary_control_type == 'net_v':
                x_1 = x[:,0]
                x_2 = x[:,1]
                return  (-np.pi/2-torch.atan(-x_1/x_2))*y.squeeze(1)
        else:
            if self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]      
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2      
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r) *y.squeeze(1) + self.base_function(x)


    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()

class ResNet_Tanh_base(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube',base_function=None):
        super(ResNet_Tanh_base, self).__init__()
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
            elif self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2
                mask = x_1 > -1/2
                base = torch.ones_like(x_1)
                base[mask] = 0
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r)*y.squeeze(1)+base

            elif self.boundary_control_type == 'net_v':
                x_1 = x[:,0]
                x_2 = x[:,1]
                return  (-np.pi/2-torch.atan(-x_1/x_2))*y.squeeze(1)
        else:
            if self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]      
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2      
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r) *y.squeeze(1) + self.base_function(x)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
class ResNet_ST_base(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube',base_function=None):
        super(ResNet_ST_base, self).__init__()
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
        
        if self.base_function == None:
            if self.boundary_control_type == 'none':
                return y.squeeze(1)
            elif self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2
                mask = x_1 > -1/2
                base = torch.ones_like(x_1)
                base[mask] = 0
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r)*y.squeeze(1)+base

            elif self.boundary_control_type == 'net_v':
                x_1 = x[:,0]
                x_2 = x[:,1]
                return  (-np.pi/2-torch.atan(-x_1/x_2))*y.squeeze(1)
        else:
            if self.boundary_control_type == 'net_u':
                x_1 = x[:,0]
                x_2 = x[:,1]      
                r = (x[:,0] ** 2+ x[:,1] **2) **1/2      
                return (np.pi/2-torch.atan(-x_1/x_2))* torch.sin(np.pi/2*r) *y.squeeze(1) + self.base_function(x)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
