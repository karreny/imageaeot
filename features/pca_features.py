import torch
from sklearn.decomposition import IncrementalPCA

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

