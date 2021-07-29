from nets import *

import numpy as np
import matplotlib.pyplot as plt
import torch 

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

r = np.arange(0, 1, 0.01)
theta = np.arange(0, np.pi/2, 0.01)

R,THETA =np.meshgrid(r, theta)
fig = plt.figure(figsize=(5,4))

X = R*np.cos(THETA)
Y = R*np.sin(THETA)

bd_X = np.cos(np.linspace(0,np.pi/2,100))
bd_Y = np.sin(np.linspace(0,np.pi/2,100))

def func(r,theta,d,net,net_like=True,find_slide=None):
    size = list(r.shape)[0]
    size2 = list(theta.shape)[1]
    point_batch = np.stack((r,theta),axis=2).reshape(size*size2,2)
    if d>2:
        point_batch = np.hstack((np.zeros([size*size,d-2]),point_batch))
    if find_slide!= None:
        point_batch[:,find_slide[0]] = find_slide[1]
    if net_like == True:
        point_batch_x = point_batch[:,0] * np.cos(point_batch[:,1])
        point_batch_y = point_batch[:,0] * np.sin(point_batch[:,1])
        point_batch = np.stack((point_batch_x,point_batch_y),axis=1)
        print(point_batch.shape)
        net_value_batch = net.predict(point_batch)
        return net_value_batch.reshape(size,size2)
    else:
        net_value_batch = net(point_batch)
        return net_value_batch.reshape(size,size2)

def true_solution(x_batch):
    r = x_batch[:,0]
    theta = x_batch[:,1]
    mask = r > 1/2
    z = np.zeros_like(r)
    z[mask] = 1
    z[~mask] = 0
    return z

def base_function(x_batch):
    r = x_batch[:,0]
    theta = x_batch[:,1]
    mask = r * np.cos(theta) + r * np.sin(theta) > 1/2
    z = np.zeros_like(r)
    z[mask] = 1
    z[~mask] = 0
    return z

load_data = torch.load('slu_7_29_11_37.t7')
net_u = load_data['net_u']
net_v = load_data['net_v']

Z_true = func(R,THETA,2,true_solution,net_like=False)
Z_predict = func(R,THETA,2,net_u,net_like=True)
Z_test = func(R,THETA,2,net_v,net_like=True)

plt.plot(bd_X,bd_Y,'k')
plt.contourf(X,Y,abs(Z_true-Z_predict),cmap='hot', alpha = 0.8,levels = 20)
plt.title('error mesh(PINN, auto bdry)')
plt.colorbar()
plt.savefig('slu_7_29_11_37.png',bbox_inches='tight')
plt.show()