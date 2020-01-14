import numpy as np
import os

from .utils import load_metadata, load_features
from .eval_utils import generate_feature_splits, compute_pred_labels, compute_confusion_mat

def eval_ot_matrix(args, logger):
    metadata = load_metadata(args.metafile)
    features = load_features(args.featfile)
    ot_mat = np.load(os.path.join(args.save_dir, "ot_%s_%s_reg_%s.npy" % (args.label1, args.label2, args.reg)))

    # generate data matrices
    feat1 = features[get_label_indices(df=metadata, label=args.label1)]
    feat2 = features[get_label_indices(df=metadata, label=args.label2)]

    labels1 = generate_feature_splits(feats1, nbins=args.nbins)
    labels2 = generate_feature_splits(feats2, nbins=args.nbins)

    pred1 = compute_pred_labels(ot_mat=ot_mat, ref_labels=labels2, num_labels=args.nbins)
    pred2 = compute_pred_labels(ot_mat=ot_mat, ref_labels=labels1, num_labels=args.nbins)

    conf1 = compute_confusion_mat(pred_labels=pred1, ref_labels=labels1, num_labels=args.nbins)
    conf2 = compute_confusion_mat(pred_labels=pred2, ref_labels=labels2, num_labels=args.nbins)

    savefile1 = os.path.join(args.save_dir, "conf1_%s_%s_reg_%s.npy" % (args.label1, args.label2, args.reg))
    savefile2 = os.path.join(args.save_dir, "conf2_%s_%s_reg_%s.npy" % (args.label1, args.label2, args.reg))
    
    np.save(savefile1, conf1)
    np.save(savefile2, conf2)

    logger.info("Confusion matrix 1: %s", % str(conf1))
    logger.info("Confusion matrix 2: %s", % str(conf2))
