from sklearn.metrics import pairwise
import numpy as np
import pandas as pd
import ot

def get_label_indices(df, label):
    return np.array(df.index[df['label'] == label].tolist())

def load_metadata(fname):
    return pd.read_csv(fname)

def load_features(fname):
    return np.loadtxt(fname)

def compute_distance_mat(f1, f2):
    return pairwise.pairwise_distances(f1,f2)

def sinkhorn_ot(dist_mat, reg=1):
    '''Computes regularized OT using sinkhorn algorithm'''
    a = np.ones(dist_mat.shape[0])/dist_mat.shape[0]
    b = np.ones(dist_mat.shape[1])/dist_mat.shape[1]
    M = dist_mat
    return ot.sinkhorn(a,b,M,reg)
