from pickle import NONE
import torch
import torch.nn as nn

import numpy as np

from settings import *
from utils import *
from nets import *

import time
import os

from hyperopt import fmin, tpe, hp

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
param_dict_default = {
    'm_u':50, # width of solution network
    'm_v':150, # width of test function network
    'lr0_u':1e-3, # initial learning rate of solution optimizer
    'lr0_v':5e-3, # initial learning rate of test function optimizer
    'nu_u':'None', # adjust rate for lr_u
    'nu_v':'None',  # adjust rate for lr_v, for scheme, see paper.
    'net_u':'ResNet_Relu', # the model of slu net, see file nets.py
    'net_v':'ResNet_Tanh', # the model of test net
    'alg':'Friedrich', # Friedrich/PINN
    'restart_time': None, # iters to rebuild the net
    'sampling_type':'lhs', # npr/lhs : np.random/Latin Hypercube Sampling
    'optimizer_u':'Adam', # optimizer for slu net Adam/Rms
    'optimizer_v':'Rmsprop', # optimizer for test net Adam/Rms
    'boundary_control_type':'Auto' # ways of PINN handling the boundary, default autometic, can be L2 
}

def train(param_dict):
    # basic params
    d = 2
    m_u = param_dict['m_u']
    m_v = param_dict['m_v']
    alg =  param_dict['alg']
    sampling_type = param_dict['sampling_type']

    # hyper params
    lr0_u = param_dict['lr0_u']
    lr0_v = param_dict['lr0_v']
    nu_u = param_dict['nu_u']
    nu_v = param_dict['nu_v']

    # domains
    domain_shape = 'fan' # the shape of domain  
    domain_intervals = domain_parameter(d)

    # experiment params
    n_epoch = 50001 # number of outer iterations
    N_inside_train = 45000 # number of training points inside domain
    N_boundary_train = 5000 # number of training points on boundary
    N_test = 10000 # number of testing points

    # optionals 
    restart_time = param_dict['restart_time']
    flag_diri_boundary_term = True  
    n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
    n_update_each_batch_test = 1

    # interface params
    verbose = True

    n_epoch_show_info = 200 # epochs to show info of experiments
    n_epoch_save_slu_plot = 1000
    n_epoch_save_loss_plot = 10000

    flag_save_loss_plot = True
    flag_save_slu_plot = False # if show plot during the training
    flag_output_results = True # if save the results as files in current directory

    print('Train start! Setting Params:',param_dict)
    #----------------------------------------------

    lr_u_seq,lr_v_seq = generate_lr_scheme(n_epoch,lr0_u,lr0_v,nu_u,nu_v,alg=param_dict['alg'],restart_time=restart_time)

    torch.manual_seed(0)
    np.random.seed(0)
    
    if alg == 'Friedrich' and restart_time != None:
        net_u = generate_network(param_dict['net_u'],d,m_u//2,net_type = 'net_u',boundary_control_type=param_dict['boundary_control_type'])
    else:
        net_u = generate_network(param_dict['net_u'],d,m_u,net_type = 'net_u',boundary_control_type=param_dict['boundary_control_type'])
    net_v = generate_network(param_dict['net_v'],d,m_v,net_type = 'net_v',boundary_control_type='auto')

    net_u,net_v = init_net(net_u),init_net(net_v)
    
    # root
    state = {'net_u': net_u,'net_v':net_v}
    if os.path.exists('../result/checkpoint') == False:
        os.mkdir('../result/checkpoint')
    if os.path.exists('../result/image') == False:
        os.mkdir('../result/image')
    if os.path.exists('../result/log') == False:
        os.mkdir('../result/log')
    if os.path.exists('../result/data') == False:
        os.mkdir('../result/data')

    torch.save(state, '../result/checkpoint/initial.t7')

    # manuel_load 
    # load_data = torch.load('./checkpoint/initial.t7')
    # net_u = load_data['net_u']
    # net_v = load_data['net_v']
    #----------------------------------------------

    optimizer_u = generate_optimizer(net_u,param_dict['optimizer_u'],lr0_u)
    optimizer_v = generate_optimizer(net_v,param_dict['optimizer_v'],lr0_v)

    l2errorseq = np.zeros((n_epoch,))
    maxerrorseq = np.zeros((n_epoch,))

    x_test = generate_uniform_points_in_fan(domain_intervals,N_test,0,np.pi/2)

    # -------------------- Training -------------------------

    n_iter = 0

    norm_vec_batch = norm_vec(N_boundary_train,d)  # multiply by 2 since we will discard half of them

    bestl2error = 1
    bestmaxerror = 1
    start_time = time.time()

    PINN_lossseq= np.zeros((n_epoch,))

    while n_iter < n_epoch:

        if restart_time!=None:
            if n_iter == restart_time:

                for param in net_u.named_parameters():
                    param[1].requires_grad = False

                net_u = generate_network(param_dict['net_u'],d,m_u,net_type='net_u',boundary_control_type = param_dict['boundary_control_type'],base_function = net_u) 
                net_u = init_net(net_u)

                optimizer_u = generate_optimizer(net_u,param_dict['optimizer_u'], lr_u_seq[n_iter])
                
                # for param in net_u.named_parameters():
                #     print(param[0],param[1].requires_grad)

                # print('net_u',net_u)


        ## generate training and testing data (the shape is (N,d)) or (N,d+1) 

        for param_group in optimizer_u.param_groups:
            param_group['lr'] = lr_u_seq[n_iter]
        
        for param_group in optimizer_v.param_groups:
            param_group['lr'] = lr_v_seq[n_iter]  

        ## Train the solution net
  
        net_u.train()
        net_v.train()

        for p in net_u.parameters():
            p.requires_grad = True

        for p in net_v.parameters():
            p.requires_grad = False

        for i in range(n_update_each_batch):
            x1_train = data_to_cuda(generate_uniform_points_in_fan(domain_intervals,N_inside_train,0,np.pi/2))

            if flag_diri_boundary_term == True:
                x2_train = data_to_cuda(generate_uniform_points_on_fan(domain_intervals,N_boundary_train,0,np.pi/2,direction='inflow')) # multiply by 2 since we will discard half of them
  
            def closure_u():
                optimizer_u.zero_grad()
                optimizer_v.zero_grad()
                if alg == 'Friedrich':
                    loss_1 = compute_Friedrich_loss(net_u,net_v,x1_train,x2_train,norm_vec_batch)
                    if param_dict['boundary_control_type'] == 'L2':
                        loss_2 = compute_boundary_loss(net_u, x2_train)
                        loss_1 += 1000 * loss_2
                    # print("loss = %.3f\n"%(loss_1),end='', flush=True)
                elif alg == 'PINN':
                    loss_1 = compute_PINN_loss(net_u,x1_train)
                    if param_dict['boundary_control_type'] == 'L2':
                        loss_2 = compute_boundary_loss(net_u, x2_train)
                        loss_1 += 1000 * loss_2
                loss_1.backward()
                PINN_lossseq[n_iter] = loss_1.item()
                return loss_1

            optimizer_u.step(closure_u)
        
        for p in net_u.parameters():
            p.requires_grad = False

        for p in net_v.parameters():
            p.requires_grad = True

        if alg == 'Friedrich':
            for i in range(n_update_each_batch_test):
                x1_train = data_to_cuda(generate_uniform_points_in_fan(domain_intervals,N_inside_train,0,np.pi/2))

                if flag_diri_boundary_term == True:
                    x2_train = data_to_cuda(generate_uniform_points_on_fan(domain_intervals,N_boundary_train,0,np.pi/2,direction='inflow'))     

                def closure_v():
                    optimizer_u.zero_grad()
                    optimizer_v.zero_grad()
                    loss_2 = -compute_Friedrich_loss(net_u,net_v,x1_train,x2_train,norm_vec_batch)
                    loss_2.backward()
                    return loss_2


            optimizer_v.step(closure_v)

        net_u.eval()
        net_v.eval()   

        l2errorseq,maxerrorseq = do_evaluate_record_error(l2errorseq,maxerrorseq,net_u,true_solution,x_test,n_iter)

        # Save the best slu_net and smallest error 
        if l2errorseq[n_iter] < bestl2error:
            state = {'net_u': net_u,'net_v':net_v,'besterror':l2errorseq[n_iter]}
            torch.save(state, '../result/checkpoint/best.t7')
            bestl2error = l2errorseq[n_iter]

        if maxerrorseq[n_iter] < bestmaxerror:
            bestmaxerror = maxerrorseq[n_iter]  

        if flag_save_slu_plot == True and n_iter % n_epoch_save_slu_plot == 0:
            do_visualize_slu(domain_intervals,net_u,net_v,true_solution,n_iter)

        if flag_save_loss_plot == True and n_iter % n_epoch_save_loss_plot == 0 and n_iter > 0:
            do_visualize_loss(l2errorseq,maxerrorseq,PINN_lossseq,param_dict,n_iter)
            
        # Show information
        if n_iter%n_epoch_show_info==0 and verbose == True:
            print("epoch = %d\n"%(n_iter),end='', flush=True)
            print("l2 error = %2.3e, max error = %2.3e, best l2 error=%2.3e, best max error=%2.3e\n" % (l2errorseq[n_iter],maxerrorseq[n_iter],bestl2error,bestmaxerror), end='', flush=True)
            print('used time=%.3f s \n'% (time.time()-start_time))
            start_time = time.time()

        n_iter = n_iter + 1
    
    localtime = time.localtime(time.time())
    time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
    state = {'net_u': net_u,'net_v':net_v,'besterror':bestl2error}
    torch.save(state, '../result/checkpoint/slu_%s.t7'%(time_text))

    # print the minimal L2 error and max error in the end 
    
    print('min l2 error =  %2.3e,  ' % min(l2errorseq), end='', flush=True)
    print('min max error =  %2.3e,  ' % min(maxerrorseq), end='', flush=True)
    
    if flag_output_results == True:
        save_data_log(d,l2errorseq,maxerrorseq,PINN_lossseq,param_dict,n_epoch,N_inside_train,N_boundary_train)
    # empty the GPU memory
    torch.cuda.empty_cache()
    

    return bestl2error


if __name__ == '__main__':
    fspace = {
        'm_u': hp.choice('m_u', [150]),
        'm_v': hp.choice('m_v', [150]),

        #'lr0_u':2e-4 * (2 + hp.randint('lr0_u',5)),
        'lr0_u': hp.choice('lr0_u', [1e-3]),
        #'lr0_v':2e-4 * (4 + hp.randint('lr0_v',8)),
        'lr0_v': hp.choice('lr0_v', [3e-3]),

        # 'nu_u':4000 * (2+ hp.randint('nu_u',2)),
        'nu_u':hp.choice('nu_u', [15000]),
        # 'nu_v':5000 * (2 + hp.randint('nu_v',2)),
        'nu_v':hp.choice('nu_v', [15000]),

        'net_u':hp.choice('net_u', ['ResNet_Tanh']),
        'net_v':hp.choice('net_v', ['ResNet_Tanh']),

        'alg':hp.choice('alg', ['PINN']),

        'restart_time': hp.choice('restart_time', [None]), # iters to rebuild the net

        'sampling_type':hp.choice('sampling_type', ['lhs']),
        # 'sampling_type':hp.choice('sampling_type', ['lhs']),

        'optimizer_u':hp.choice('optimizer_u', ['Adam']), # optimizer for slu net Adam/Rms
        'optimizer_v':hp.choice('optimizer_v', ['Rmsprop']), # optimizer for test net Adam/Rms

        'boundary_control_type':hp.choice('boundary_control_type', ['L2'])
        }

    best = fmin(
        fn=train,
        space=fspace,
        algo=tpe.suggest,
        max_evals=1)
    
    print(best)

