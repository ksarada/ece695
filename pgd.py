import torch
from utils import normalization_function
from input_gradient_calculation import inp_grad_calc
import time

class ProjectedGradientDescent:
    """
    The Projected Gradient Descent Method for adversarial attack
    This method was introduced by Madry et. al.
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """
    def __init__(self, num_iter, epsilon, eps_iter, clip_min, clip_max, targeted, rand_init, seed):
        """
        Initialize the attack parameters
        :param num_iter: number of iterations for PGD
        :param epsilon: attack strength
        :param eps_iter: attack strength per iteration
        :param clip_min, clip_max: clip the adversarial input between clip_min and clip_max
        :param targeted: True or False
        :param rand_init: 0 or 1. If 1, adds a random perturbation with strength epsilon
        to the input before performing PGD operation
        :param seed: seed used for random number generator
        """
        super(ProjectedGradientDescent, self).__init__()
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.num_iter = num_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.seed = seed
        
            
    def generate(self, x, labels, model, loss_func, mean, std):
        """
        returns the pgd adversarial output 
        
        x: The inputs to the model, x lies in the range [0,1]
        labels: Correct labels corresponding to x (untargeted) or the target label (targeted attacks)
        model: the source model for generating adversary
        loss_func: Used to calculate the error
        """
        x_orig = x.clone()
        torch.manual_seed(self.seed)
        gradVal = torch.zeros_like(x).cuda()
        #x.requires_grad_(True)
        #if self.rand_init:
        #    x = x + (2*self.epsilon)*torch.rand_like(x) - self.epsilon
        #    x = torch.clamp(x, self.clip_min, self.clip_max)
        start = time.time()
        for i in range(self.num_iter):
            # We need to normalize the input with mean and std, which was used to train the model
            #x_norm = normalization_function(x, mean, std)
            x_norm = x
            
            #print(i, end='')
            #print(' ', end='')
            
            inp_grad = inp_grad_calc(x_norm, labels, model, loss_func) # Calculate the gradient of loss w.r.t. input
            #x.requires_grad_(True)
            #if self.targeted == 'True':
            #    inp_grad = -1*inp_grad

            #input_sum = torch.sum(torch.tanh(x_norm))
            #with torch.enable_grad():
            #inp2_grad = torch.autograd.grad(input_sum, x,
            #                           retain_graph=False,
            #                           create_graph=False)[0]
            #if(i<=30):
            #self.eps_iter = 0.5/255. #self.eps_iter*0.5
            #elif(i>30):
            #    self.eps_iter = 0.25 #self.eps_iter*0.5
            #else:
            #    self.eps_iter = 0.5 #self.eps_iter*0.5
            #x.requires_grad_(False)
            #print(torch.norm(inp_grad1))
            #print(torch.norm(inp_grad2))
            #gradVal = torch.sign(inp_grad) # + 0.0001*inp2_grad) #0.99*gradVal + inp_grad/torch.mean((inp_grad), [0,1,2,3], keepdim=True)
            #inp_grad = inp_grad #1 + inp_grad2
            gradVal = 0.9*gradVal + 0.1*inp_grad #/torch.mean(torch.abs(inp_grad), dim=[0,1,2,3], keepdim=True)
            x -= self.eps_iter * gradVal #inp_grad/torch.mean((inp_grad), [0,1,2,3], keepdim=True) #torch.sign(inp_grad)
            eta = torch.clamp(x-x_orig, min=-self.epsilon, max=self.epsilon)
            x = torch.clamp(x_orig+eta, self.clip_min, self.clip_max)      
            #print(time.time() - start)  
        #print(x.shape)
        #mean_adv_distortion = torch.mean(x - x_orig, 0, True)
        #print(mean_adv_distortion.shape)        
        return x
        
        
    
        
            
        
