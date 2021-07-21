import numpy as np
import torch
import time 
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nets import *

# generate uniform distributed points in a domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_in_cube(domain_intervals,N):
    d = domain_intervals.shape[0]
    points = np.zeros((N,d))
    for i in range(d):
        points[:,i] = np.random.uniform(domain_intervals[i,0],domain_intervals[i,1],(N,))
    points = torch.Tensor(points)
    points.requires_grad=False
    return points

def generate_uniform_points_in_cylinder(domain_intervals,N):
    d = domain_intervals.shape[0]
    r = domain_intervals.shape[1]
    points = np.zeros((N,d))
    polar_axis = np.zeros((N,d))
    polar_axis[:,0] = np.random.uniform(domain_intervals[0,0],domain_intervals[0,1],(N,))
    polar_axis[:,1] = np.random.uniform(0,r**2,(N,))
    polar_axis[:,2] = np.random.uniform(0,2*np.pi,(N,))
    points[:,0] = polar_axis[:,0]
    points[:,1] = np.sqrt(polar_axis[:,1]) * np.cos(polar_axis[:,2])
    points[:,2] = np.sqrt(polar_axis[:,1]) * np.sin(polar_axis[:,2])
    points = torch.Tensor(points)
    points.requires_grad=False
    return points

def generate_uniform_points_in_circle(domain_intervals,N): # generate points at T=0 in a circle
    d = domain_intervals.shape[0]
    r = domain_intervals.shape[1]
    points = np.zeros((N,d))
    polar_axis = np.zeros((N,d))
    polar_axis[:,1] = np.random.uniform(0,r**2,(N,))
    polar_axis[:,2] = np.random.uniform(0,2*np.pi,(N,))
    points[:,1] = np.sqrt(polar_axis[:,1]) * np.cos(polar_axis[:,2])
    points[:,2] = np.sqrt(polar_axis[:,1]) * np.sin(polar_axis[:,2])
    points = torch.Tensor(points)
    points.requires_grad=False
    return points

# generate uniform distributed points in the sphere {x:|x|<R}
# input d is the dimension
def generate_uniform_points_in_sphere(d,R,N):
    points = np.random.normal(size=(N,d))
    scales = (np.random.uniform(0,R,(N,)))**(1/d)
    for i in range(N):
        points[i,:] = points[i,:]/np.sqrt(np.sum(points[i,:]**2))*scales[i]
    points = torch.Tensor(points)
    points.requires_grad=False
    return points

# generate uniform distributed points on the boundary of domain [a1,b1]X[a2,b2]X...X[ad,bd]
# domain_intervals = [[a1,b1],[a2,b2],...,[ad,bd]]
def generate_uniform_points_on_cube(domain_intervals,N_each_face):
    d = domain_intervals.shape[0]
    if d == 1:
        points = np.array([[domain_intervals[0,0]],[domain_intervals[0,1]]])
        points = torch.Tensor(points)
        points.requires_grad=False
        return points
    else:
        points = np.zeros((2*d*N_each_face,d))
        for i in range(d):
            points[2*i*N_each_face:(2*i+1)*N_each_face,:] = np.insert(generate_uniform_points_in_cube(np.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,0]*np.ones((1,N_each_face)), axis = 1)
            points[(2*i+1)*N_each_face:(2*i+2)*N_each_face,:] = np.insert(generate_uniform_points_in_cube(np.delete(domain_intervals,i,axis=0),N_each_face), i, values=domain_intervals[i,1]*np.ones((1,N_each_face)), axis = 1)
        points = torch.Tensor(points)
        points.requires_grad=False
        return points

# generate uniform distributed points on the boundary of domain {|x|<R}
def generate_uniform_points_on_sphere(d,R,N_boundary):
    if d == 1:
        points = np.array([[-R],[R]])
        points = torch.Tensor(points)
        points.requires_grad=False        
        return points
    else:
        points = np.zeros((N_boundary,d))
        for i in range(N_boundary):
            points[i,:] = np.random.normal(size=(1,d))
            points[i,:] = points[i,:]/np.sqrt(np.sum(points[i,:]**2))*R
        points = torch.Tensor(points)
        points.requires_grad=False
        return points

# discard specific boundary points when given the direction of inflow
def take_left_down_side_bd(x2_train,if_cuda=False):
    if if_cuda == False:
        N_2 = x2_train.shape[0]
        d = x2_train.shape[1]
        x2_train_selected = np.zeros([N_2//2,d])
    else:
        N_2 = x2_train.size(0)
        d = x2_train.size(1)
        x2_train_selected = torch.zeros([N_2//2,d])
    x2_train_selected[0:N_2//4,:] = x2_train[0:N_2//4,:]
    x2_train_selected[N_2//4:2*N_2//4,:] = x2_train[2*N_2//4:3*N_2//4,:]
    return x2_train_selected

def generate_lr_scheme(n_epoch,lr0_u,lr0_v,nu_u,nu_v):
    lr_u_seq = np.zeros(n_epoch)  # set the learning rates for each epoch
    lr_v_seq = np.zeros(n_epoch)

    for i in range(n_epoch):
        lr_u_seq[i] = lr0_u * (1/10) ** (i/nu_u)

    for i in range(n_epoch):
        lr_v_seq[i] = lr0_v * (1/10) ** (i/nu_v)

    return lr_u_seq,lr_v_seq

def generate_network(net_name,dim,width,boundary_control_type='None'):
    if net_name == 'ResNet_Relu':
        net = ResNet_Relu(width,dim,boundary_control_type=boundary_control_type)
    elif net_name == 'ResNet_Tanh':
        net = ResNet_Tanh(width,dim,boundary_control_type=boundary_control_type)  
    elif net_name == 'ResNet_Tanh':
        net = ResNet_Tanh(width,dim,boundary_control_type=boundary_control_type)  
        
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return net


# evaluate relative l2 error and max error
def evaluate_rel_l2_error(model, true_solution,x_batch): 
        l2error = np.sqrt(sum((model.predict(x_batch) - true_solution(x_batch))**2)/x_batch.shape[0])
        u_l2norm = np.sqrt(sum((true_solution(x_batch))**2)/x_batch.shape[0])
        return l2error/u_l2norm

def evaluate_rel_max_error(model, true_solution,x_batch):
    maxerror = np.max(np.absolute(model.predict(x_batch) - true_solution(x_batch)))
    u_maxnorm = np.max(np.absolute(true_solution(x_batch)))
    return maxerror/u_maxnorm

def evaluate_l2_error(model, true_solution,x_batch):
    l2error = np.sqrt(sum((model.predict(x_batch) - true_solution(x_batch))**2)/x_batch.shape[0])
    return l2error

def evaluate_max_error(model,true_solution, x_batch):
    maxerror = np.max(np.absolute(model.predict(x_batch) - true_solution(x_batch)))
    return maxerror

# evaluate and record error
def do_evaluate_record_error(l2errorseq,maxerrorseq,net_u,true_slu,x_test,n_iter):
    l2error = evaluate_rel_l2_error(net_u, x_test)
    l2errorseq[n_iter] = l2error
    maxerror = evaluate_rel_max_error(net_u, x_test)
    maxerrorseq[n_iter] = maxerror
    return l2errorseq,maxerrorseq 

# compute the value of slu net on the points we use in the plot
def compute_net_value_on_plots(x,y,d,net,net_like=True,find_slide=None):  # find_slide argument takes the form (a,b), a means the value of a-th dim is fixed to be b.
    size = list(x.shape)[0]
    point_batch = np.stack((x,y),axis=2).reshape(size*size,2)
    if d>2:
        point_batch = np.hstack((point_batch,np.zeros([size*size,d-2])))
    if find_slide!= None:
        point_batch[:,find_slide[0]] = find_slide[1]
    if net_like == True:
        net_value_batch = net.predict(point_batch)
        return net_value_batch.reshape(size,size)
    else:
        net_value_batch = net(point_batch)
        return net_value_batch.reshape(size,size)

# visualize slu net and test net and save fig
def do_visualize_slu(domain_intervals,net_u,net_v,true_solution,n_iter):
    d = domain_intervals.shape[0]
    x_plot = np.arange(domain_intervals[0,0], domain_intervals[0,1]+0.01, 0.02)
    y_plot = np.arange(domain_intervals[1,0], domain_intervals[1,1]+0.01, 0.02)
    x, y = np.meshgrid(x_plot, y_plot) 

    # 3D plot
    # print('plotting surface')
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.clear()
    # z = func(x,y,d,net_u)
    # ax.plot_surface(x,y,z,rstride=1, cstride=1,cmap='rainbow',alpha = 0.5)
    # z = func(x,y,d,true_solution,net_like=False)
    # ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='rainbow',alpha = 0.5)
    # plt.savefig("./image/slu_func_%d.png"%(k))
    # plt.close()

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.clear()
    # z = func(x,y,d,net_v)
    # ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='hot',alpha = 0.5)
    # plt.savefig("./image/phi_func_%d.png"%(k))
    # plt.close()

    # 2D plot
    print('plotting contour')
    fig = plt.figure()
    z_1 = compute_net_value_on_plots(x,y,d,net_u)
    z_2 = compute_net_value_on_plots(x,y,d,true_solution,net_like=False)
    plt.contourf(x,y,z_1-z_2,cmap='RdYlGn', alpha = 0.8,levels = 20,vmin=-0.2,vmax=0.2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar()
    plt.savefig("./image/slu_func_%d.png"%(n_iter))
    plt.close()

    fig = plt.figure()
    z = compute_net_value_on_plots(x,y,d,net_v)
    plt.contourf(x,y,z,cmap='RdYlGn', alpha = 0.8,levels = 20)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.colorbar()
    plt.savefig("./image/phi_func_%d.png"%(n_iter))
    plt.close()
    return 0

# visualize the loss and save fig
def do_visualize_loss(l2errorseq,maxerrorseq,n_iter):
    fig = plt.figure()
    plt.title('l2 error and max error')
    plt.plot(l2errorseq[0:n_iter],label='l2 error')
    plt.plot(maxerrorseq[0:n_iter],label = 'max error')
    plt.legend()
    plt.yscale('log')
    plt.grid()
    plt.savefig("./image/errorseq_%d.png"%(n_iter))
    plt.close()
    return 0 

# log
def save_data_log(l2errorseq,maxerrorseq,param_dict,n_epoch,N_inside_train,N_boundary_train,restart_period):
    localtime = time.localtime(time.time())
    time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
    filename = 'result_'+str(param_dict['d'])+'d_'+time_text+'.data'
    lossseq_and_errorseq = np.zeros((2,n_epoch))
    lossseq_and_errorseq[0,:] = l2errorseq
    lossseq_and_errorseq[1,:] = maxerrorseq

    f = open('./data/'+filename, 'wb')
    pickle.dump(lossseq_and_errorseq, f)
    f.close()
        
    #save parameters
    text = 'Parameters:\n'
    text = text + 'd = ' + str(param_dict['d']) +'\n'
    text = text + 'm_u = ' + str(param_dict['m_u']) +'\n'
    text = text + 'm_v = ' + str(param_dict['m_v']) +'\n'
    text = text + 'n_epoch = ' + str(n_epoch) +'\n'
    text = text + 'N_inside_train = ' + str(N_inside_train) +'\n'
    text = text + 'N_boundary_train = ' + str(N_boundary_train) +'\n'
    text = text + 'initial slu net lr  = ' + str(param_dict['lr0_u']) +'\n'
    text = text + 'initial test net lr  = ' + str(param_dict['lr0_v']) +'\n'
    text = text + 'nu_u = ' + str(param_dict['nu_u']) +'\n'
    text = text + 'nu_v = ' + str(param_dict['nu_v']) +'\n'
    text = text + 'restart_period = ' + str(restart_period) +'\n'

    text = text + 'min l2 error = ' + str(min(l2errorseq)) + ', '
    text = text + 'min max error = ' + str(min(maxerrorseq)) + ', '

    with open('./log/'+'Parameters_'+time_text+'.log','w') as f:   
        f.write(text)  
    
    return 0
    