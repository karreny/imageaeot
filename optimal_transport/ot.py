import numpy as np
import os

from .utils import load_metadata, load_features, get_label_indices, compute_distance_mat, sinkhorn_ot

def get_distance_mat(args, logger):

    savefile = os.path.join(args.save_dir, "distmat_%s_%s.npy" % (args.label1, args.label2))

    # load distance matrix if it already exists
    if os.path.isfile(savefile):
        logger.info("Distance matrix loaded from %s" % savefile)
        return np.load(savefile)

    # otherwise compute it
    # load data files
    metadata = load_metadata(args.metafile)
    features = load_features(args.featfile)

    # generate data matrices
    indices1 = get_label_indices(df=metadata, label=args.label1)
    indices2 = get_label_indices(df=metadata, label=args.label2)

    feat1 = features[indices1]
    feat2 = features[indices2]

    distmat = compute_distance_mat(feat1, feat2)
    np.save(savefile, distmat)
    logger.info("Distance matrix saved at %s" % savefile)
    return distmat

def get_ot_matrix(args, logger):
    
    savefile = os.path.join(args.save_dir, "ot_%s_%s_reg_%s.npy" % (args.label1, args.label2, args.reg))
    
    # return if ot matrix already exists
    if os.path.isfile(savefile):
        logger.info("OT matrix at %s already exists" % savefile)
        return

    dist_mat = get_distance_mat(args, logger)
    logger.info("Distance matrix shape is %s" % str(dist_mat.shape))
    
    ot_mat = sinkhorn_ot(dist_mat, reg=args.reg)
    
    np.save(savefile, ot_mat)
    logger.info("OT matrix saved at %s" % savefile)
