import torch
import numpy as np
import torch.nn.functional as F

def inp_grad_calc(x, labels, model, loss_func):
        """
        Calculates the gradient of error with respect to input (del_loss/del_input)
        :param x: input
        :param labels: The labels used to calculate the loss or error
        :param model: Source model
        :loss_func: Loss function used to calculate the error or loss
        """
        inputs = x.clone().detach()
        inputs.requires_grad_(True)
        inputs.grad = None       
        if model.type=='SNN': #module.type=='SNN':
            #----------------------------------------------------------------------------
            # preprocessing: conv1 weight matrix rotated by 180
            ############################################################################
            #print(model.features)
            out_channel = model.features.module[0].weight.size()[0] # 64
            in_channel = model.features.module[0].weight.size()[1] # 3
            weight_new = np.zeros((model.features.module[0].weight.size()))
            for i in range(out_channel):
                for j in range(in_channel):
                    weight_new[i,j,0,0:] = np.flip(model.features.module[0].weight.detach().cpu().numpy()[i,j,2,0:])
                    weight_new[i,j,1,0:] = np.flip(model.features.module[0].weight.detach().cpu().numpy()[i,j,1,0:])
                    weight_new[i,j,2,0:] = np.flip(model.features.module[0].weight.detach().cpu().numpy()[i,j,0,0:])
            weight_new = np.transpose(weight_new,(1,0,2,3))
            weight_rotate = torch.from_numpy(weight_new).float().cuda()
            
            #output, input_spike_count, grad_mem_conv1, total_sc = model(inputs,0, True)
            output, input_spike_count, grad_mem_conv1, obj_count, total_spike_count, input_sc = model(inputs,0, 40, True)       
            #output, input_spike_count, grad_mem_conv1 = model(inputs,0, 10, True)
            output = output/model.timesteps
        #elif model.module.type == 'ANN':
        else:
            ##output, act_conv, _ = model(inputs)
            output = model(inputs)
            #print(output)
            #act_conv.detach_()

        
        #print(torch.sum(torch.abs(input_sc)).item(), end='')
        #print('  ', end='')
        #tsc = torch.zeros(1).cuda()
        #tscc = torch.zeros(1).cuda()
        lIn = 0
        tsc = 0
        #lWt = [2., 2.,1.,1., 1., 1.] 
        lWt = [2., 4.,4.,8., 8., 8., 8., 1.,1. , 1.] 
        #lWt_a = [0., 4.,4.,8., 8., 8., 8., 1.,1.] 
        #lWt = [1., 2., 2., 4.,4., 4., 8., 8., 8., 8., 8.,8., 1., 1., 1.] 
        
        for k,v in total_spike_count.items(): #obj_count.items():
            #tsc = tsc + torch.sum(v)
            #tscc = tscc + torch.sum(v)
            
            if(len(v.shape)>2):
                total_neurons = v.size(1)*v.size(2)*v.size(3)
            else:
                total_neurons = v.size(1)
                #print(' {} {} {} '.format(v.size(1), v.size(2), v.size(3)), end='')
                #tsc = tsc + torch.pow(torch.sum(v),2) #norm(torch.pow(v, 2.), p=3)*lWt[lIn]
            #if(lIn<4):
            #tsc = tsc + torch.norm(torch.pow(v, 2.), p=3)*lWt[lIn] #torch.sum(v)*lWt[lIn]
                #tsc = tsc + torch.sigmoid(torch.sum(torch.sigmoid(v/100000))/total_neurons)
            tsc = tsc - torch.sum((v-80.)**2.)*lWt[lIn]  #**3.)
            #tsc = tsc - ((torch.sum(v/100.) - total_neurons)**2.)*lWt[lIn] #torch.sqrt(v)) #*torch.sum(v) #*lWt[lIn]
            #tsc = tsc + (torch.sum(1./(100.-v)**2.)) #- total_neurons)**2. 
            #tsc = tsc - (torch.sum((v - 100)**2)  - total_neurons)**2
            #tsc = tsc - (torch.sum((v - 100.)**2.)) #  - total_neurons)**2.
            #tsc = tsc + torch.sum(torch.tanh(v/500)) #- total_neurons)**2
            #if(lIn<4):
            #    spike_proxy = torch.tanh(v/1e9)
            #relu = torch.nn.ReLU()
                
            #print(lIn) 
            lIn = lIn + 1  

        #tisc = tsc + torch.sum(torch.abs(input_sc))
        tlsc = tsc + torch.sum(inputs) #*4. #input_sc**2) #torch.abs(input_sc))
        tisc = torch.sum(torch.abs(input_sc))
        ta_isc = torch.sum(torch.abs(input_sc))
            
      
        
        error = -1.*tlsc + 0.01*loss_func(output, labels) #(-1.*tlsc) #+  0.01*loss_func(output, labels) #-1.*tlsc - 1.5*1e7*loss_func(output, labels) # #torch.exp((tscc/1e4)) #tsc #loss_func(output, labels)
        
        error.backward()        
        if model.type=='SNN': #module.type=='SNN':
            grad_mem_conv1 = model.grad_mem1 #module.grad_mem1
            # del_loss/del_input = conv(del_loss/del_conv1, W_conv1(180rotated))
            inp_grad = F.conv2d(grad_mem_conv1, weight_rotate, padding=1)
        else: #lif model.module.type=='ANN':
            inp_grad = inputs.grad.data
        # reset        
        #inputs.requires_grad_(False)
       
        inputs.requires_grad_(False)
        inputs.grad = None
        model.zero_grad() #model.module.zero_grad()
        
        return inp_grad #, inp_grad_2
