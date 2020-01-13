import torch

import numpy as np
import os

def extract_AE_features(dataloader, net, savefile):
    '''Extract data features from given model'''
    
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net.cuda()

    net.eval()
    for sample in dataloader:
        inputs = sample['image']

        if is_cuda:
            inputs = inputs.cuda()

        outputs, _ = net.encode(inputs)

        with open(savefile, 'ab') as f:
    	    np.savetxt(f, outputs.cpu().data, fmt='%f')

