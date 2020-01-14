import numpy as np
from scipy import stats

import os

def compute_pred_labels(ot_mat, ref_labels, num_labels):
    pred_labels = []

    for l in range(num_labels):
        pred = np.matmul(ot_mat, (ref_labels == l)*1.)
        pred_labels.append(pred)

    return np.stack(pred_labels, axis=-1)

def compute_confusion_mat(pred_labels, ref_labels, num_labels):
    confusion_mat = []

    for l in range(num_labels):
        conf = pred_labels[ref_labels==l]
        conf = np.sum(conf, axis=0)
        confusion_mat.append(conf)

    return np.stack(confusion_mat, axis=1)

def generate_feature_splits(feats, nbins):
    _, _, indices = stats.binned_statistic(feats, feats, nbins)
    return indices
