import torch
import torch.nn as nn

import numpy as np

from settings import *
from utils import *
from nets import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

param_dict = {
    'd':2, # dimension of problem
    'm_u':50, # width of solution network
    'm_v':150, # width of test function network
    'lr0_u':3e-4, # initial learning rate of solution optimizer
    'lr0_v':3e-3, # initial learning rate of test function optimizer
    'nu_u':1e4, # adjust rate for lr_u
    'nu_v':1e4,  # adjust rate for lr_v, for scheme, see paper.
    'net_u':'ResNet_Relu', # the model of slu net, see file nets.py
    'net_v':'ResNet_Relu' # the model of test net
}

def train(param_dict):
    # basic params
    d = param_dict['d']
    m_u = param_dict['m_u']
    m_v = param_dict['m_v']
    network_file = param_dict['net']

    # hyper params
    lr0_u = param_dict['lr0_u']
    lr0_v = param_dict['lr0_v']
    nu_u = param_dict['nu_u']
    nu_v = param_dict['nu_v']

    # domains
    domain_shape = 'cube' # the shape of domain  
    domain_intervals = domain_parameter(d)

    # experiment params
    n_epoch = 5000 # number of outer iterations
    N_inside_train = 90000 # number of training points inside domain
    N_boundary_train = 10000 # number of training points on boundary
    N_test = 10000 # number of testing points

    # optionals 
    restart_period = 1000000
    flag_diri_boundary_term = True  
    sample_times = 1
    n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
    n_update_each_batch_test = 1

    # interface params
    n_epoch_show_info = 20 # epochs to show info of experiments
    n_epoch_save_slu_plot = 100
    n_epoch_save_loss_plot = 500

    flag_save_loss_plot = True
    flag_save_slu_plot = False # if show plot during the training
    flag_output_results = True # if save the results as files in current directory

    print('Train start! initial slu net lr = %.2e, initial test net lr = %.2e, nu_u = %d, nu_v = %d'%(lr0_u,lr0_v,nu_u,nu_v))
    #----------------------------------------------

    lr_u_seq,lr_v_seq = generate_lr_scheme(n_epoch,lr0_u,lr0_v,nu_u,nu_v)

    torch.manual_seed(0)
    np.random.seed(0)

    net_u = generate_network(param_dict['net_u'],d,m_u,boundary_control_type='net_u')
    net_v = generate_network(param_dict['net_v'],d,m_v,boundary_control_type='net_v')

    state = {'net_u': net_u,'net_v':net_v}
    torch.save(state, './checkpoint/initial.t7')

    # manuel_load 
    # load_data = torch.load('./checkpoint/initial.t7')
    # net_u = load_data['net_u']
    # net_v = load_data['net_v']
    #----------------------------------------------

    optimizer_u = torch.optim.Adam(net_u.parameters(),lr= lr0_u)
    optimizer_v = torch.optim.RMSprop(net_v.parameters(),lr=lr0_v)

    l2errorseq = np.zeros((n_epoch,))
    maxerrorseq = np.zeros((n_epoch,))

    x_test = generate_uniform_points_in_cube(domain_intervals,N_test)     

    # -------------------- Training -------------------------

    n_iter = 0

    norm_vec_batch = norm_vec(2*N_boundary_train,d,'cube')  # multiply by 2 since we will discard half of them
    norm_vec_batch = take_left_down_side_bd(norm_vec_batch,if_cuda=True)

    bestl2error = 1

    while n_iter < n_epoch:
        start_time = time.time()
        if (n_iter % restart_period == 0) and (n_iter>1) :
            net_v = generate_network(param_dict['net_v'],d,m_v,boundary_control_type='net_v')
            optimizer_v = torch.optim.Adam(net_v.parameters(),lr=lr_v_seq[n_iter])

        ## generate training and testing data (the shape is (N,d)) or (N,d+1) 

        for param_group in optimizer_u.param_groups:
            param_group['lr'] = lr_u_seq[n_iter]
        
        for param_group in optimizer_v.param_groups:
            param_group['lr'] = lr_v_seq[n_iter]  

        ## Train the solution net
  
        net_u.train()
        net_v.train()

        for i in range(n_update_each_batch):
            x1_train = generate_uniform_points_in_cube(domain_intervals,N_inside_train)

            if flag_diri_boundary_term == True:
                x2_train = generate_uniform_points_on_cube(domain_intervals,2*N_boundary_train//4) # multiply by 2 since we will discard half of them
                x2_train = take_left_down_side_bd(x2_train,if_cuda=True)

            ## Compute the loss  
            Aphi_prod, phi_norm = compute_Aphi_prod_phi_norm_autograd(x1_train,x2_train,net_u,net_v,norm_vec_batch)  # apply algorithms

            loss_1 = Aphi_prod**2/phi_norm

            ### Update the network
            optimizer_u.zero_grad()
            loss_1.backward()
            optimizer_u.step()
    
        for i in range(n_update_each_batch_test):
            x1_train = generate_uniform_points_in_cube(domain_intervals,N_inside_train)

            if flag_diri_boundary_term == True:
                x2_train = generate_uniform_points_on_cube(domain_intervals,2*N_boundary_train//4)       
                x2_train = take_left_down_side_bd(x2_train,if_cuda=True)

            ## Compute the loss  
            Aphi_prod, phi_norm = compute_Aphi_prod_phi_norm_autograd(x1_train,x2_train,net_u,net_v,norm_vec_batch)

            loss_2 = -Aphi_prod**2/phi_norm

            ### Update the network
            optimizer_v.zero_grad()
            loss_2.backward()
            optimizer_v.step()

        net_u.eval()
        net_v.eval()   

        l2errorseq,maxerrorseq = do_evaluate_record_error(l2errorseq,maxerrorseq,net_u,true_solution,x_test,n_iter)

        # Save the best slu_net and smallest error 
        if l2errorseq[n_iter] < bestl2error:
            state = {'net_u': net_u,'net_v':net_v,'besterror':l2errorseq[n_iter]}
            torch.save(state, './checkpoint/best.t7')
            bestl2error = l2errorseq[n_iter]

        if flag_save_slu_plot == True and n_iter % n_epoch_save_slu_plot == 0:
            do_visualize_slu(domain_intervals,net_u,net_v,true_solution,n_iter)

        if flag_save_loss_plot == True and n_iter % n_epoch_save_loss_plot == 0 and n_iter > 0:
            do_visualize_loss(d,l2errorseq,maxerrorseq)
            
        # Show information
        if n_iter%n_epoch_show_info==0:
            print("epoch = %d, loss1 = %2.5f, loss2 = %2.5f, loss1-loss2 = %2.5f" %(n_iter,loss_1.item(),-loss_2.item(),loss_1.item()+loss_2.item()), end='', flush=True)
            print("l2 error = %2.3e, max error = %2.3e, best l2 error=%2.3e" % (l2errorseq[n_iter],maxerrorseq[n_iter],bestl2error), end='', flush=True)
            print('used time=%.3f s \n'% (time.time()-start_time))

        n_iter = n_iter + 1
        

    # print the minimal L2 error and max error in the end 
    print('min l2 error =  %2.3e,  ' % min(l2errorseq), end='', flush=True)
    print('min max error =  %2.3e,  ' % min(maxerrorseq), end='', flush=True)
    
    if flag_output_results == True:
        save_data_log(l2errorseq,maxerrorseq,param_dict,n_epoch,N_inside_train,N_boundary_train,restart_period)

    # empty the GPU memory
    torch.cuda.empty_cache()

    return bestl2error


if __name__ == '__main__':
    train()

