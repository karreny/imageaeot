import torch
from sklearn.decomposition import IncrementalPCA

from .utils import save_img

import numpy as np
import os

def extract_PCA_features(dataloader, savefile, net=None, n_components=128):
    '''Extract data features from given model'''

    # setup PCA model
    model = IncrementalPCA(n_components=n_components)
    
    # fit PCA model
    for sample in dataloader:
        inputs = sample['image']
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1) # flatten images
        if batch_size >= n_components:
            model.partial_fit(inputs.numpy())
        else:
            print("batch size %s is smaller than n_components %s" % (batch_size, n_components))

    # get projections
    for sample in dataloader:
        inputs = sample['image']
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1) # flatten images 

        outputs = model.transform(inputs.numpy())

        with open(savefile, 'ab') as f:
    	    np.savetxt(f, outputs, fmt='%f')

def visualize_PCA_recon(dataloader, dataloader_bs1, savedir, net=None, n_components=128):
    '''saves example reconstruction to save directory'''
    
    # setup PCA model
    model = IncrementalPCA(n_components=n_components)
    
    # fit PCA model
    for sample in dataloader:
        inputs = sample['image']
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1) # flatten images
        if batch_size >= n_components:
            model.partial_fit(inputs.numpy())
        else:
            print("batch size %s is smaller than n_components %s" % (batch_size, n_components))

    # get projections
    for idx, sample in enumerate(dataloader_bs1):
        inputs = sample['image']
        inputs = inputs.view(1, -1) # flatten image

        outputs = model.inverse_transform(model.transform(inputs.numpy()))

        save_img(inputs, os.path.join(savedir, "input_%s.png" % idx))
        save_img(torch.from_numpy(outputs), os.path.join(savedir, "recon_%s.png" % idx))

        if idx >= 5:
            break
