import torch

import numpy as np
import os

def extract_flattened_features(dataloader, savefile, net=None, n_components=None):
    '''Extract data features from given model'''

    for sample in dataloader:
        inputs = sample['image']
        batch_size = len(inputs)
        inputs = inputs.view(batch_size, -1) # flatten images 

        outputs = inputs.numpy()

        with open(savefile, 'ab') as f:
    	    np.savetxt(f, outputs, fmt='%f')

