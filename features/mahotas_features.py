import torch
import mahotas.features as mf

import numpy as np
import os

def extract_mahotas_features(dataloader, savefile, net=None, n_components=None):
    '''Extract data features from given model'''

    # build feature dictionary
    feature_dict = {'eccentriciy': mf.eccentricity, 
                    'ellipse_axes': mf.ellipse_axes,
                    'roundness': mf.roundness,
                    'haralick': mf.haralick,
                    'tas': mf.tas,
                    }

    features = {k:[] for k in feature_dict.keys()}

    # extract image features
    for sample in dataloader:
        inputs = sample['image']
        batch_size = len(inputs)
        assert(batch_size == 1)
        inputs = inputs.view(64, 64).numpy()

        # convert to int
        inputs = np.rint(inputs*255)
        inputs = inputs.astype(int)

        for k in feature_dict.keys():
            features[k].append(feature_dict[k](inputs))

    features = {k: np.array(features[k]) for k in features.keys()}
    for k in features.keys():
        with open("%s_%s.txt" % (savefile, k), 'wb') as f:
            feats_to_save = np.array(features[k])
            batch_size = feats_to_save.shape[0]
            feats_to_save = np.reshape(feats_to_save, (batch_size, -1))
            np.savetxt(f, feats_to_save, fmt='%f')
