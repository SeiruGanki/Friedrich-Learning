import math
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# torch.set_default_tensor_type('torch.DoubleTensor')

degree = 1
c = 0.1
bd = np.pi
"""
define neutral network 
"""
class TanhResNet_para(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(TanhResNet_para, self).__init__()
        self.fc11 = nn.Linear(dim, m)
        self.fc12 = nn.Linear(m, m)
        self.fc13 = nn.Linear(m, m)
        self.fc14 = nn.Linear(m, m)
        self.fc15 = nn.Linear(m, m)
        self.fc16 = nn.Linear(m, m)

        self.fc21 = nn.Linear(dim, m)
        self.fc22 = nn.Linear(m, m)
        self.fc23 = nn.Linear(m, m)
        self.fc24 = nn.Linear(m, m)
        self.fc25 = nn.Linear(m, m)
        self.fc26 = nn.Linear(m, m)

        self.fc31 = nn.Linear(dim, m)
        self.fc32 = nn.Linear(m, m)
        self.fc33 = nn.Linear(m, m)
        self.fc34 = nn.Linear(m, m)
        self.fc35 = nn.Linear(m, m)
        self.fc36 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        Ix[2,2] = 1. 
        self.Ix = Ix
        self.dim =dim
        self.outlayer1 = nn.Linear(m, 1, bias = False)
        self.outlayer2 = nn.Linear(m, 1, bias = False)
        self.outlayer3 = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc11(x)
        y = torch.tanh(y)
        y = self.fc12(y)
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc13(y)     
        y = torch.tanh(y)
        y = self.fc14(y)       
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc15(y)      
        y = torch.tanh(y)
        y = self.fc16(y)    
        y = torch.tanh(y)
        y = y+s
                
        y = self.outlayer1(y)

        if self.boundary_control_type == 'none':
            y1 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,0]*(bd-x[:,0])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,0]*(bd-x[:,0])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])).unsqueeze(1)
            y1 = y * factor
        ######

        s = x@self.Ix
        y = self.fc21(x)
        y = torch.tanh(y)
        y = self.fc22(y)
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc23(y)     
        y = torch.tanh(y)
        y = self.fc24(y)       
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc25(y)      
        y = torch.tanh(y)
        y = self.fc26(y)    
        y = torch.tanh(y)
        y = y+s
                
        y = self.outlayer2(y)

        if self.boundary_control_type == 'none':
            y2 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,1]*(bd-x[:,1])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,1]*(bd-x[:,1])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])).unsqueeze(1)
            y2 = y * factor
        ########

        s = x@self.Ix
        y = self.fc31(x)
        y = torch.tanh(y)
        y = self.fc32(y)
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc33(y)     
        y = torch.tanh(y)
        y = self.fc34(y)       
        y = torch.tanh(y)
        y = y+s
        
        s=y
        y = self.fc35(y)      
        y = torch.tanh(y)
        y = self.fc36(y)    
        y = torch.tanh(y)
        y = y+s
                
        y = self.outlayer3(y)

        if self.boundary_control_type == 'none':
            y3 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,2]*(bd-x[:,2])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,2]*(bd-x[:,2])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1])).unsqueeze(1)
            y3 = y * factor        

        return torch.cat((y1,y2,y3),dim=1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
    
# class ReLUResNet(nn.Module):
#     def __init__(self, m,dim=2,degree=1,boundary_control_type='cube'):
#         super(ReLUResNet_para, self).__init__()
#         self.fc1 = nn.Linear(dim, m)
#         self.fc2 = nn.Linear(m, m)
#         self.fc3 = nn.Linear(m, m)
#         self.fc4 = nn.Linear(m, m)
#         self.fc5 = nn.Linear(m, m)
#         self.fc6 = nn.Linear(m, m)

#         self.boundary_control_type = boundary_control_type
#         self.outlayer = nn.Linear(m, 3, bias = False)
#         self.degree = degree
#         self.dim =dim
#         Ix = torch.zeros([dim,m]).cuda()
#         Ix[0,0] = 1.
#         Ix[1,1] = 1. 
#         Ix[2,2] = 1. 
#         self.Ix = Ix

#     def forward(self, x):
#         s = x@self.Ix
#         y = self.fc1(x)
#         y = F.relu(y**self.degree)
#         y = self.fc2(y)
#         y = F.relu(y**self.degree)
#         y = y+s
        
#         s=y
#         y = self.fc3(y)     
#         y = F.relu(y**self.degree)
#         y = self.fc4(y)       
#         y = F.relu(y**self.degree)
#         y = y+s
        
#         s=y
#         y = self.fc5(y)      
#         y = F.relu(y**self.degree)
#         y = self.fc6(y)    
#         y = F.relu(y**self.degree)
#         y = y+s

#         y = self.outlayer(y)


#         if self.boundary_control_type == 'none':
#             return y

#         elif self.boundary_control_type == 'H_net':
#             factor = torch.cat(((x[:,0]*(bd-x[:,0])),(x[:,1]*(bd-x[:,1])),(x[:,2]*(bd-x[:,2]))),dim=1)
#             return y * factor

#         elif self.boundary_control_type == 'E_net':
#             factor = torch.cat(((x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])),(x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])),(x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1]))),dim=1)
#             return y * factor

#         elif self.boundary_control_type == 'phiH_net':
#             factor = torch.cat(((x[:,0]*(bd-x[:,0])),(x[:,1]*(bd-x[:,1])),(x[:,2]*(bd-x[:,2]))),dim=1)
#             return y * factor

#         elif self.boundary_control_type == 'phiE_net':
#             factor = torch.cat(((x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])),(x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])),(x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1]))),dim=1)
#             return y * factor


#     def predict(self, x_batch):
#         tensor_x_batch = torch.Tensor(x_batch)
#         tensor_x_batch.requires_grad=False
#         y = self.forward(tensor_x_batch)
#         return y.cpu().detach().numpy()

class ReLUResNet_para(nn.Module):  
    def __init__(self, m,dim=2,boundary_control_type='cube'):
        super(ReLUResNet_para, self).__init__()
        self.fc11 = nn.Linear(dim, m)
        self.fc12 = nn.Linear(m, m)
        self.fc13 = nn.Linear(m, m)
        self.fc14 = nn.Linear(m, m)
        self.fc15 = nn.Linear(m, m)
        self.fc16 = nn.Linear(m, m)

        self.fc21 = nn.Linear(dim, m)
        self.fc22 = nn.Linear(m, m)
        self.fc23 = nn.Linear(m, m)
        self.fc24 = nn.Linear(m, m)
        self.fc25 = nn.Linear(m, m)
        self.fc26 = nn.Linear(m, m)

        self.fc31 = nn.Linear(dim, m)
        self.fc32 = nn.Linear(m, m)
        self.fc33 = nn.Linear(m, m)
        self.fc34 = nn.Linear(m, m)
        self.fc35 = nn.Linear(m, m)
        self.fc36 = nn.Linear(m, m)

        self.boundary_control_type = boundary_control_type
        Ix = torch.zeros([dim,m]).cuda()
        Ix[0,0] = 1.
        Ix[1,1] = 1. 
        Ix[2,2] = 1. 
        self.Ix = Ix
        self.dim =dim
        self.outlayer1 = nn.Linear(m, 1, bias = False)
        self.outlayer2 = nn.Linear(m, 1, bias = False)
        self.outlayer3 = nn.Linear(m, 1, bias = False)

    def forward(self, x):
        s = x@self.Ix
        y = self.fc11(x)
        y = torch.relu(y)
        y = self.fc12(y)
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc13(y)     
        y = torch.relu(y)
        y = self.fc14(y)       
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc15(y)      
        y = torch.relu(y)
        y = self.fc16(y)    
        y = torch.relu(y)
        y = y+s
                
        y = self.outlayer1(y)

        if self.boundary_control_type == 'none':
            y1 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,0]*(bd-x[:,0])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,0]*(bd-x[:,0])).unsqueeze(1)
            y1 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,1]*x[:,2]*(bd-x[:,1])*(bd-x[:,2])).unsqueeze(1)
            y1 = y * factor
        ######

        s = x@self.Ix
        y = self.fc21(x)
        y = torch.relu(y)
        y = self.fc22(y)
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc23(y)     
        y = torch.relu(y)
        y = self.fc24(y)       
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc25(y)      
        y = torch.relu(y)
        y = self.fc26(y)    
        y = torch.relu(y)
        y = y+s
                
        y = self.outlayer2(y)

        if self.boundary_control_type == 'none':
            y2 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,1]*(bd-x[:,1])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,1]*(bd-x[:,1])).unsqueeze(1)
            y2 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,2]*x[:,0]*(bd-x[:,2])*(bd-x[:,0])).unsqueeze(1)
            y2 = y * factor
        ########

        s = x@self.Ix
        y = self.fc31(x)
        y = torch.relu(y)
        y = self.fc32(y)
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc33(y)     
        y = torch.relu(y)
        y = self.fc34(y)       
        y = torch.relu(y)
        y = y+s
        
        s=y
        y = self.fc35(y)      
        y = torch.relu(y)
        y = self.fc36(y)    
        y = torch.relu(y)
        y = y+s
                
        y = self.outlayer3(y)

        if self.boundary_control_type == 'none':
            y3 = y

        elif self.boundary_control_type == 'H_net':
            factor = (x[:,2]*(bd-x[:,2])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'E_net':
            factor = (x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'phiH_net':
            factor = (x[:,2]*(bd-x[:,2])).unsqueeze(1)
            y3 = y * factor

        elif self.boundary_control_type == 'phiE_net':
            factor = (x[:,0]*x[:,1]*(bd-x[:,0])*(bd-x[:,1])).unsqueeze(1)
            y3 = y * factor        

        return torch.cat((y1,y2,y3),dim=1)

    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()