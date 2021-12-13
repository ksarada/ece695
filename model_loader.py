import os
import torch
import torch.nn as nn
import nn_models as vgg
import resnet_cifar_1 as resnet18 
import vgg_cifar as vgg_c
import resnet_im as resnet_im
import mobilenet_im as mv2


# map between model name and function
models = {
    'ann'              : vgg.ANN_VGG,
    'ann_source'       : vgg.ANN_VGG,
    'ann_target'       : vgg.ANN_VGG,
    'snnconv'          : vgg.SNN_VGG,
    'snnbp'            : vgg.SNN_VGG,
}

def load(model_name, batch_size, model_file_name='', snn_thresholds=[20.76, 2.69, 2.33, 0.28, 1.14]):
    """
    Creates an instance of the model, initializes and loads the trained model parameters
    :param model_name: ann, snnconv or snnbp
    :param batch_size: bath size of the input
    :param model_file_name: location and name of the trained parameter file
    """
    if model_name=='ann_target':
        #net = models[model_name](vgg_name = 'VGG16', labels = 10)
        #net = resnet18.resnet18_c1()
        #net = vgg_c.vgg16_bn()
        net = resnet_im.resnet50()
        #net = mv2.mobilenet_v2()
    elif model_name=='ann_source':
        #net = models[model_name](vgg_name = 'VGG11', labels = 10)
        #net = vgg_c.vgg9()
        net = resnet_im.resnet18()
    elif model_name=='ann':
        net = models[model_name](vgg_name = 'VGG11', labels = 10)
        #net = vgg_c.vgg9()
        #net = resnet_im.resnet18()
    elif model_name == 'snnconv':
        net = models[model_name](batch_size = batch_size, vgg_name = 'VGG11', activation = 'Linear', labels=10, timesteps=100, leak_mem=1.0)
        net.threshold_init(scaling_threshold=1.0, reset_threshold=0, thresholds = snn_thresholds[:], default_threshold=1.0)    
    elif model_name == 'snnbp':
        net = models[model_name](batch_size = batch_size, vgg_name = 'VGG11', activation = 'Linear', labels=10, timesteps=100, leak_mem=1.) #0.99
        net.threshold_init(scaling_threshold=1., reset_threshold=0, thresholds = snn_thresholds[:], default_threshold=1.0)    #0.7

    #net = torch.nn.DataParallel(net.cuda())
    net.features = torch.nn.DataParallel(net.features)
    net.cuda()
    model_file = model_file_name
    assert os.path.exists(model_file), model_file + " does not exist."
    stored = torch.load(model_file, map_location=lambda storage, loc: storage)
#    torch.save(stored['state_dict'],'snnbp_checkpoint1.pt')

    cur_dict = net.state_dict() 

    #for k, v in cur_dict.items():
    #    print(k)

    #for k, v in stored['state_dict'].items():
    #    print(k)
    #thresh_ = [4.4, 1.75, 0.79, 0.69, 0.14, 0.21, 0.22, 0.62, 0.99]
    if(model_name=='snnbp' or model_name=='ann'): #(0)
        print(model_name)
        kIn = 0
        lIn = 0
        for k, v in stored['state_dict'].items():
        #if(kIn<14):
            #print(k)
            if(k.find('weight')!=-1):
                jIn = 0
                for k1, v1 in cur_dict.items():
                    if(jIn==kIn):
                        cur_dict[k1] = nn.Parameter(stored['state_dict'][k].data)
                        lIn = lIn + 1
                        #print(kIn,end='')
                        #print(jIn,end='')
                        #print(k1,end='')
                        #print(' ', end="")
                        #print(k)
                    jIn = jIn + 1
                kIn = kIn + 1
    
        net.load_state_dict(cur_dict)
    #else:
    #    net.load_state_dict(stored['state_dict'])

    return net
