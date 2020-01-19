import torch

from .utils import save_img

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

        #outputs, _ = net.encode(inputs)
        outputs = net.encode(inputs)

        with open(savefile, 'ab') as f:
    	    np.savetxt(f, outputs.cpu().data, fmt='%f')

def visualize_AE_recon(dataloader, net, savedir):
    '''saves example reconstruction to save directory'''

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net.cuda()

    net.eval()

    for idx, sample in enumerate(dataloader):
        inputs = sample['image']

        if is_cuda:
            inputs = inputs.cuda()
        
        outputs = net.decode(net.encode(inputs))
        save_img(inputs, os.path.join(savedir, "input_%s.png" % idx))
        save_img(outputs, os.path.join(savedir, "recon_%s.png" % idx))

        if idx >= 5:
            break
