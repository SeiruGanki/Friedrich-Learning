import torch
from torch import Tensor, optim
import numpy as np
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid, log
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_in_cube_time_dependent,\
    generate_uniform_points_on_cube, generate_uniform_points_on_cube_time_dependent,\
    generate_uniform_points_in_sphere, generate_uniform_points_in_sphere_time_dependent,\
    generate_uniform_points_on_sphere, generate_uniform_points_on_sphere_time_dependent,\
    generate_learning_rates,generate_uniform_points_in_cylinder,generate_uniform_points_in_circle

from network import ReLUResNet_para,TanhResNet_para

from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import matplotlib.pyplot as plt
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D

from solution_maxwell_auto import *

import torch.nn as nn
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

def train(lr_u, lr_v, lr_u2,lr_v2,exp_bench_u, exp_bench_v,exp_bench_u2,exp_bench_v2):
    lr_u = 3e-6   
    lr_v = 3e-3

    exp_bench_u = 8000
    exp_bench_v = 15000

    
    ########### basic parameters #############
    d = 3  # dimension of problem
    m_u = 250  # number of nodes in each layer of solution network
    m_v = 50
    mu = 1
    sigma = 1
    domain_shape = 'cube' ## the shape of domain 
    domain_intervals = domain_parameter(d)

    loss_type = 'frac'
    flag_adjust_decaying = False
    flag_quadradic_loss = False
    flag_part_loss = True
    flag_diri_boundary_term = True
    #----------------------------------------------

    ########### used hyper-parameters #############
    n_epoch = 20000 # number of outer iterations
    N_inside_train = 50000 # number of training sampling points inside the domain in each epoch (batch size)
    N_boundary_train = 0
    restart_period = 500000
    

   
    matrix_A = torch.zeros(3,3,3)
    matrix_A[0,1,2] = -1
    matrix_A[0,2,1] = 1 
    matrix_A[1,0,2] = 1
    matrix_A[1,2,0] = -1 
    matrix_A[2,0,1] = -1
    matrix_A[2,1,0] = 1 
    
    A_batch = torch.zeros(N_inside_train,3,3,3)
    A_batch[:] = matrix_A
    A_batch = A_batch

    sample_times = 1
    n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
    n_update_each_batch_test = 1

    lr_u_seq = np.zeros(n_epoch)  # set the learning rates for each epoch
    lr_v_seq = np.zeros(n_epoch)

    for i in range(n_epoch):
        lr_u_seq[i] = lr_u * (1/10) ** (i/exp_bench_u)

    for i in range(n_epoch):
        lr_v_seq[i] = lr_v * (1/10) ** (i/exp_bench_v)

    print('Train start! lr_u = %.2e, lr_v = %.2e, exp_bench_u = %d, exp_bench_v = %d,restart_period=%d'%(lr_u,lr_v,exp_bench_u,exp_bench_v,restart_period))
    print(flag_quadradic_loss)
    para = 1 # the infinitesimal in ln-like loss function

    #----------------------------------------------


    ########### unused hyper-parameters, may be unchanged #############
    lambda_1 = 1
    activation = 'ReLU3'  # activation function for the solution net

    flag_preiteration_by_small_lr = False  # If pre iteration by small learning rates
    lr_pre = 1e-4  # learning rates in preiteration
    n_update_each_batch_pre = 20 # number of iterations in each epoch in preiteration
    #----------------------------------------------

    ########### Interface parameters #############
    n_epoch_show_info = 20 # max([round(n_epoch/100),1]) # how many epoch will it show information
    N_test = 10000 # number of testing points

    flag_l2error = True
    flag_maxerror = True
    flag_givenpts_l2error = False #在给定的点上估计l2 error，这里看起来是那个固定的原点
    flag_givenpts_maxerror = False #
    
    if flag_givenpts_l2error == True or flag_givenpts_maxerror == True:
        given_pts = zeros((1,d)) #

    flag_savefig = True
    flag_show_plot = False # if show plot during the training
    flag_output_results = True # if save the results as files in current directory
    #----------------------------------------------

        
    ########### Depending parameters #############
    # u_net = network_file.network(d,m, activation_type = activation, boundary_control_type = 'homo_unit_%s'%(domain_shape)) #这个网络是原函数u的模型网络
    # v_net = network_file.network(d,m, activation_type = activation, boundary_control_type = 'homo_unit_%s'%(domain_shape)) #这个网络是weak sense \phi 的网络，我应该不齐
    torch.manual_seed(0)
    np.random.seed(0)
    H_net = ReLUResNet_para(m_u,dim=d,boundary_control_type = 'H_net') 
    E_net = ReLUResNet_para(m_u,dim=d,boundary_control_type = 'E_net') 
    phiH_net = TanhResNet_para(m_v,dim=d,boundary_control_type = 'phiH_net') 
    phiE_net = TanhResNet_para(m_v,dim=d,boundary_control_type = 'phiE_net') 

    nets = [H_net,E_net,phiH_net,phiE_net]
    nets_name = ['H_net','E_net','phiH_net','phiE_net']
    for net in nets:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    state = {'H_net': H_net,'E_net':E_net,'phiH_net':phiH_net,'phiE_net':phiE_net}
    torch.save(state, './checkpoint/initial.t7')
    #----------------------------------------------

    ######### the fairness #############
    # load_data = torch.load('./checkpoint/start_load.t7')
    # H_net = load_data['H_net']
    # E_net = load_data['E_net']
    # phiH_net = load_data['phiH_net']
    # phiE_net = load_data['phiE_net']
    # print(load_data)
    #torch.manual_seed(0)
    #np.random.seed(0)
    #----------------------------------------------


    ########### decide the number of points in training boundary #############
    if flag_diri_boundary_term:
        if domain_shape == 'cube':
            if d == 1:
                N_each_face_train = 1
            else:
                N_each_face_train = max([1,int(round(N_inside_train/2/d))]) # number of sampling points on each domain face when training
            N_boundary_train = 2*d*N_each_face_train
        elif domain_shape == 'sphere':
            if d == 1:
                N_boundary_train = 2
            else:
                N_boundary_train = N_inside_train # number of sampling points on each domain face when training
    else:
        N_boundary_train = 0
    #----------------------------------------------
    ########### some functions #############
    # function to evaluate the discrete L2 error (input x_batch is a 2d numpy array; output is a scalar Tensor)
    def evaluate_rel_l2_error(H_net,E_net,x_batch): #relative error
        l2error_H = sum((H_net.predict(x_batch) - true_solution_H(x_batch))**2)/x_batch.shape[0]
        l2error_E = sum((E_net.predict(x_batch) - true_solution_E(x_batch))**2)/x_batch.shape[0]
        H_l2norm = sum((true_solution_H(x_batch))**2)/x_batch.shape[0]
        E_l2norm = sum((true_solution_E(x_batch))**2)/x_batch.shape[0]
        return sqrt(l2error_H+l2error_E)/sqrt(H_l2norm+E_l2norm)

    def evaluate_rel_max_error(H_net,E_net,x_batch):
        maxerror_H = np.max(absolute(H_net.predict(x_batch) - true_solution_H(x_batch)))
        maxerror_E = np.max(absolute(E_net.predict(x_batch) - true_solution_E(x_batch)))

        H_maxnorm = np.max(absolute(true_solution_H(x_batch)))
        E_maxnorm = np.max(absolute(true_solution_E(x_batch)))
        return max(maxerror_H,maxerror_E)/max(H_maxnorm,E_maxnorm)

    def evaluate_l2_error(H_net,E_net,x_batch): #relative error
        l2error_H = sum((H_net.predict(x_batch) - true_solution_H(x_batch))**2)/x_batch.shape[0]
        l2error_E = sum((E_net.predict(x_batch) - true_solution_E(x_batch))**2)/x_batch.shape[0]
        return sqrt((l2error_H+l2error_E)/6)

    def evaluate_max_error(H_net,E_net,x_batch):
        maxerror_H = np.max(absolute(H_net.predict(x_batch) - true_solution_H(x_batch)))
        maxerror_E = np.max(absolute(E_net.predict(x_batch) - true_solution_E(x_batch)))
        return max(maxerror_H,maxerror_E)

    def func(x,y,d,net,net_like=True,find_slide=None):
        size = list(x.shape)[0]
        point_batch = np.stack((x,y),axis=2).reshape(size*size,2)
        if d>2:
            point_batch = np.hstack((point_batch,np.zeros([size*size,d-2])))
        if find_slide!= None:
            point_batch[:,find_slide[0]] = find_slide[1]
        if net_like == True:
            net_value_batch = net.predict(point_batch)[:,0]
            return net_value_batch.reshape(size,size)
        else:
            net_value_batch = net(point_batch)[:,0]
            return net_value_batch.reshape(size,size)
    #----------------------------------------------


    #################### Start ######################
    optimizer_H = optim.Adam(H_net.parameters(),lr= lr_u)
    optimizer_E = optim.Adam(E_net.parameters(),lr= lr_u)
    optimizer_phiH = optim.RMSprop(phiH_net.parameters(),lr= lr_v)
    optimizer_phiE = optim.RMSprop(phiE_net.parameters(),lr= lr_v)
    optimizers = [optimizer_H,optimizer_E,optimizer_phiH,optimizer_phiE]

    lossseq = zeros((n_epoch,))
    l2errorseq = zeros((n_epoch,))
    maxerrorseq = zeros((n_epoch,))
    givenpts_l2errorseq = zeros((n_epoch,))  # 在给定的点集上面算test error
    givenpts_maxerrorseq = zeros((n_epoch,))

    x_test = generate_uniform_points_in_cube(domain_intervals,N_test)     

    # Training
    k = 0

    x_plot = np.arange(0, np.pi, 0.05)
    y_plot = np.arange(0, np.pi, 0.05)
    x, y = np.meshgrid(x_plot, y_plot)

    bestl2error = 1

    while k < n_epoch:
        start_time = time.time()

        if flag_adjust_decaying == True:
            if k == n_epoch//3:
                for i in range(k,n_epoch):
                    lr_u_seq[i] = optimizer_H.param_groups[0]['lr'] * (1/10) ** ((i-k)/3/exp_bench_u)

                for i in range(k,n_epoch):
                    lr_v_seq[i] = optimizer_phiH.param_groups[0]['lr'] * (1/10) ** ((i-k)/3/exp_bench_v)
            if k == 2*n_epoch//3:
                for i in range(k,n_epoch):
                    lr_u_seq[i] = optimizer_H.param_groups[0]['lr'] * (1/10) ** ((i-k)/9/exp_bench_u)

                for i in range(k,n_epoch):
                    lr_v_seq[i] = optimizer_phiH.param_groups[0]['lr'] * (1/10) ** ((i-k)/9/exp_bench_v)


        if (k % restart_period == 0) and (k>1) :
            phiH_net = TanhResNet_para(m_v,dim=d,boundary_control_type = 'phiH_net') 
            phiE_net = TanhResNet_para(m_v,dim=d,boundary_control_type = 'phiE_net') 
            optimizer_phiH = optim.RMSprop(phiH_net.parameters(),lr= lr_v_seq[k])
            optimizer_phiE = optim.RMSprop(phiE_net.parameters(),lr= lr_v_seq[k])

        ## generate training and testing data (the shape is (N,d)) or (N,d+1) 
        ## label 1 is for the points inside the domain, 2 is for those on the bondary or at the initial time
        tensor_x1_train,tensor_x2_train= None,None
        if domain_shape == 'cube':
            tensor_x1_train_list = []
            for i in range(sample_times):
                x1_train = generate_uniform_points_in_cube(domain_intervals,N_inside_train)
                tensor_x1_train = Tensor(x1_train)
                tensor_x1_train.requires_grad=False
                tensor_x1_train_list.append(tensor_x1_train)

            tensor_x2_train_list = []
            if flag_diri_boundary_term == True:
                for i in range(sample_times):
                    x2_train = generate_uniform_points_on_cube(domain_intervals,N_boundary_train//4)       
                    tensor_x2_train = Tensor(x2_train)
                    tensor_x2_train.requires_grad=False
                    tensor_x2_train_list.append(tensor_x2_train)
                
        ## Set learning rate
        for param_group in optimizer_E.param_groups:
           if flag_preiteration_by_small_lr == True and k == 0:
               param_group['lr'] = lr_pre
           else:
               param_group['lr'] = lr_u_seq[k]

        for param_group in optimizer_H.param_groups:
           if flag_preiteration_by_small_lr == True and k == 0:
               param_group['lr'] = lr_pre
           else:
               param_group['lr'] = lr_u_seq[k]  

        for param_group in optimizer_phiE.param_groups:
           if flag_preiteration_by_small_lr == True and k == 0:
               param_group['lr'] = lr_pre
           else:
               param_group['lr'] = lr_v_seq[k]

        for param_group in optimizer_phiH.param_groups:
           if flag_preiteration_by_small_lr == True and k == 0:
               param_group['lr'] = lr_pre
           else:
               param_group['lr'] = lr_v_seq[k]  

        ## Train the solution net
        if flag_preiteration_by_small_lr == True and k == 0:
            temp = n_update_each_batch_pre
        else:
            temp = n_update_each_batch  # k=0, pretrain iter

        for net in nets:
            net.train()

        for i in range(n_update_each_batch):
            x1_batch = tensor_x1_train_list[i]
            x2_batch = tensor_x2_train_list[i]

            loss_1 = 0

            loss1,loss2,loss3= 0,0,0

            Aphi_prod1,Aphi_prod2,phi_norm= compute_Aphi_prod_phi_norm_autograd(x1_batch,mu,sigma,H_net,E_net,phiH_net,phiE_net,A_batch)
            
            if loss_type == 'log':
                loss1 = torch.log(Aphi_prod1**2+Aphi_prod2**2+ para * phi_norm)
                loss2 = torch.log(phi_norm)

                if flag_diri_boundary_term:
                    loss3 = 0

                loss_1 += 1* (loss1 - loss2 - lambda_1 * loss3 )

            elif loss_type == 'frac':
                if flag_quadradic_loss == False:
                    loss_1 = (Aphi_prod1+Aphi_prod2)**2/phi_norm
                else:
                    loss_1 = (Aphi_prod1**2+Aphi_prod2**2)/phi_norm       
            ### Update the network
            optimizer_E.zero_grad()
            optimizer_H.zero_grad()
            loss_1.backward()
            optimizer_E.step()
            optimizer_H.step()
    
        for i in range(n_update_each_batch_test):
            x1_batch = tensor_x1_train_list[i]
            x2_batch = tensor_x2_train_list[i]
            loss_2 = 0

            loss1,loss2,loss3= 0,0,0
            ## Compute the loss  
            Aphi_prod1,Aphi_prod2,phi_norm = compute_Aphi_prod_phi_norm_autograd(x1_batch,mu,sigma,H_net,E_net,phiH_net,phiE_net,A_batch)

            if loss_type == 'log':
                loss1 = torch.log(Aphi_prod1**2+Aphi_prod2**2+ para * phi_norm)
                loss2 = torch.log(phi_norm)

                if flag_diri_boundary_term:
                    loss3 = 0

                loss_2 += -1* (loss1 - loss2 - lambda_1 * loss3 )
            elif loss_type == 'frac':
                if flag_quadradic_loss == False:
                    loss_2 = -(Aphi_prod1+Aphi_prod2)**2/phi_norm
                else:
                    loss_2 = -(Aphi_prod1**2+Aphi_prod2**2)/phi_norm  

            optimizer_phiE.zero_grad()
            optimizer_phiH.zero_grad()
            loss_2.backward()
            optimizer_phiE.step()
            optimizer_phiH.step()

        for net in nets:
            net.eval()

        if flag_show_plot == True:
            if k%100==0:

                # Plot the suface of slu_net and phi_net
                print('plotting surface')
                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.clear()
                # z = func(x,y,d,u_net)
                # ax.plot_surface(x,y,z,rstride=1, cstride=1,cmap='rainbow',alpha = 0.5)
                # z = func(x,y,d,true_solution,net_like=False)
                # ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='rainbow',alpha = 0.5)
                # plt.savefig("./image/slu_func_%d.png"%(k))
                # plt.close()

                # fig = plt.figure()
                # ax = Axes3D(fig)
                # ax.clear()
                # z = func(x,y,d,v_net)
                # ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='hot',alpha = 0.5)
                # plt.savefig("./image/phi_func_%d.png"%(k))
                # plt.close()
                
                for slide in np.linspace(0,np.pi,11):
                    fig = plt.figure()
                    z_1 = func(x,y,d,H_net,find_slide=(2,slide))
                    z_2 = func(x,y,d,true_solution_H,net_like=False,find_slide=(2,slide))
                    plt.contourf(x,y,z_1,cmap='RdYlGn', alpha = 0.8,levels = 20,vmin=-1,vmax=1)
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                    plt.colorbar()
                    plt.savefig("./image/H_%d_%.1f.png"%(k,slide))
                    plt.close()

                for slide in np.linspace(0,np.pi,11):
                    fig = plt.figure()
                    z_1 = func(x,y,d,E_net,find_slide=(2,slide))
                    z_2 = func(x,y,d,true_solution_E,net_like=False,find_slide=(2,slide))
                    plt.contourf(x,y,z_1,cmap='RdYlGn', alpha = 0.8,levels = 20,vmin=-1,vmax=1)
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                    plt.colorbar()
                    plt.savefig("./image/E_%d_%.1f.png"%(k,slide))
                    plt.close()

                for slide in np.linspace(0,np.pi,11):
                    fig = plt.figure()
                    z = func(x,y,d,phiH_net,find_slide=(2,slide))
                    plt.contourf(x,y,z,cmap='RdYlGn', alpha = 0.8,levels = 20)
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                    plt.colorbar()
                    plt.savefig("./image/phiH_%d_%.1f.png"%(k,slide))
                    plt.close()

                for slide in np.linspace(0,np.pi,11):
                    fig = plt.figure()
                    z = func(x,y,d,phiE_net,find_slide=(2,slide))
                    plt.contourf(x,y,z,cmap='RdYlGn', alpha = 0.8,levels = 20)
                    plt.xlabel('$x_1$')
                    plt.ylabel('$x_2$')
                    plt.colorbar()
                    plt.savefig("./image/phiE_%d_%.1f.png"%(k,slide))
                    plt.close()
   
        # Save loss and L2 error
        lossseq[k] = loss_1.item()
        if flag_l2error == True:
            l2error = evaluate_rel_l2_error(H_net,E_net, x_test)
            l2errorseq[k] = l2error
        if flag_maxerror == True:
            maxerror = evaluate_rel_max_error(H_net,E_net, x_test)
            maxerrorseq[k] = maxerror

        
        # Save the best slu_net and lowest error 
        if l2errorseq[k] < bestl2error:
            state = {'H_net': H_net,'E_net':E_net,'phiH_net':phiH_net,'phiE_net':phiE_net,'besterror':l2errorseq[k]}
            torch.save(state, './checkpoint/best.t7')
            bestl2error = l2errorseq[k]

        # Save the plot of error seq alongside the training.
        if flag_savefig == True and k > 0 and k % 500 == 0:
            fig = plt.figure()
            plt.title('l2 error and max error')
            plt.plot(l2errorseq[0:k],label='l2 error')
            plt.plot(maxerrorseq[0:k],label = 'max error')
            plt.grid()
            plt.legend()
            plt.yscale('log')
            # plt.ylim((5e-2,2.5e-1))
            plt.grid()
            plt.savefig("./image/errorseq_%d.png"%(k))
            #plt.show()
            plt.close()

        # Show information
        if k%n_epoch_show_info==0:
            print("epoch = %d" %(k))
            print(", loss1 = %2.5f, loss2 = %2.5f, loss1-loss2 = %2.5f" %(loss_1.item(),-loss_2.item(),loss_1.item()+loss_2.item()), 'lr=' ,lr_u_seq[k],end='', flush=True)
            print('')

            if flag_l2error == True:
                print("l2 error = %2.3e" % l2error, end='', flush=True)
            if flag_maxerror == True:
                print(", max error = %2.3e" % maxerror, end='', flush=True)
            if flag_givenpts_l2error == True:
                print(", givenpts l2 error = %2.3e" % givenpts_l2error, end='', flush=True)
            if flag_givenpts_maxerror == True:
                print(", givenpts max error = %2.3e" % givenpts_maxerror, end='', flush=True)
            print(', best l2 error=%2.3e'%(bestl2error))
            print('time=%.3f s'% (time.time()-start_time))
            print("\n")
        # print('l2error=',l2error,'maxerror=',maxerror)
        # print('loss_1',loss_1,'loss_2',loss_2)
        # Unused pre-iteration process 
        #scheduler_u.step(bestl2error)
        #scheduler_u2.step(bestl2error)
        #scheduler_v.step(bestl2error)
        #scheduler_v2.step(bestl2error)
        # if k > 2000:
        #     scheduler_u.patience = 100
        #     scheduler_v.patience = 100

        if flag_preiteration_by_small_lr == True and k == 0:
            flag_start_normal_training = True
            if flag_l2error == True and l2error>0.9:
                flag_start_normal_training = False
            if flag_maxerror == True and maxerror>0.9:
                flag_start_normal_training = False
            if flag_start_normal_training == True:
                k = 1
                print('pre_iter finished !')
        else:
            k = k + 1
        

    #print the minimal L2 error
    if flag_l2error == True:
        print('min l2 error =  %2.3e,  ' % min(l2errorseq), end='', flush=True)
    if flag_maxerror == True:
        print('min max error =  %2.3e,  ' % min(maxerrorseq), end='', flush=True)
    if flag_givenpts_l2error == True:
        print('min givenpts l2 error =  %2.3e,  ' % min(givenpts_l2errorseq), end='', flush=True)
    if flag_givenpts_maxerror == True:
        print('min givenpts max error =  %2.3e,  ' % min(givenpts_maxerrorseq), end='', flush=True)
    

    if flag_output_results == True:
        #save the data
        localtime = time.localtime(time.time())
        time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
        filename = 'result_'+str(d)+'d_'+time_text+'.data'
        lossseq_and_errorseq = zeros((5,n_epoch))
        lossseq_and_errorseq[0,:] = lossseq
        lossseq_and_errorseq[1,:] = l2errorseq
        lossseq_and_errorseq[2,:] = maxerrorseq
        lossseq_and_errorseq[3,:] = givenpts_l2errorseq
        lossseq_and_errorseq[4,:] = givenpts_maxerrorseq
        f = open('./data/'+filename, 'wb')
        pickle.dump(lossseq_and_errorseq, f)
        f.close()
        
        # load_data = torch.load('./checkpoint/best.t7')
        # u_net = load_data['u_net']
        # v_net = load_data['v_net']

        # #save the solution and error mesh
        # if d >= 2:
        #     x, y = np.meshgrid(x_plot, y_plot)
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.clear()
        #     z = func(x,y,u_net)
        #     # plot the best slu
        #     x_plot = np.arange(-1, 1.01, 0.04)
        #     y_plot = np.arange(-1, 1.01, 0.04)
        #     x, y = np.meshgrid(x_plot, y_plot)
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.clear()
        #     z = func(x,y,u_net)
        #     ax.plot_surface(x,y,z,rstride=1, cstride=1,cmap='rainbow',alpha = 0.5)
        #     z = func(x,y,true_solution,net_like=False)
        #     ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap='rainbow',alpha = 0.5)
        #     plt.savefig("./image/best")
        #     plt.close()
        
        #save parameters
        text = 'Parameters:\n'
        text = text + 'd = ' + str(d) +'\n'
        text = text + 'm_u = ' + str(m_u) +'\n'
        text = text + 'm_v = ' + str(m_v) +'\n'
        text = text + 'n_epoch = ' + str(n_epoch) +'\n'
        text = text + 'N_inside_train = ' + str(N_inside_train) +'\n'
        text = text + 'N_boundary_train = ' + str(N_boundary_train) +'\n'
        text = text + 'lr_u_seq[0] = ' + str(lr_u_seq[0]) +'\n'
        text = text + 'lr_u_seq[-1] = ' + str(lr_u_seq[-1]) +'\n'
        text = text + 'lr_u_seq[0] = ' + str(lr_v_seq[0]) +'\n'
        text = text + 'lr_u_seq[-1] = ' + str(lr_v_seq[-1]) +'\n'
        # text = text + 'frac_bench_u = ' + str(frac_bench_u) +'\n'
        # text = text + 'frac_bench_v = ' + str(frac_bench_v) +'\n'
        text = text + 'exp_bench_u = ' + str(exp_bench_u) +'\n'
        text = text + 'exp_bench_v = ' + str(exp_bench_v) +'\n'
        text = text + 'restart_period = ' + str(restart_period) +'\n'

        if flag_l2error == True:
            text = text + 'min l2 error = ' + str(min(l2errorseq)) + ', '
        if flag_maxerror == True:
            text = text + 'min max error = ' + str(min(maxerrorseq)) + ', '
        if flag_givenpts_l2error == True:
            text = text + 'min givenpts l2 error = ' + str(min(givenpts_l2errorseq)) + ', '
        if flag_givenpts_maxerror == True:
            text = text + 'min givenpts max error = ' + str(min(givenpts_maxerrorseq)) + ', '
        with open('./log/'+'Parameters_'+time_text+'.log','w') as f:   
            f.write(text)  
        
        # loss analysis API
        # pickle_data()
        state = {'H_net': H_net,'E_net':E_net,'phiH_net':phiH_net,'phiE_net':phiE_net,'besterror':l2errorseq[k]}
        torch.save(state, './checkpoint/best_%.3f.t7'%(bestl2error))
        # empty the GPU memory
        torch.cuda.empty_cache()
        return -bestl2error


if __name__ == '__main__':
    train()

