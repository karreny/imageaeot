import numpy as np
import os

from .utils import load_metadata, load_features, get_label_indices
from .eval_utils import generate_feature_splits, compute_pred_labels, compute_roc_score

def eval_ot_matrix(args, logger):
    assert(args.nbins == 2)

    metadata = load_metadata(args.metafile)
    features = load_features(args.evalfeatfile)
    ot_mat = np.load(os.path.join(args.save_dir, "ot_%s_%s_reg_%s.npy" % (args.label1, args.label2, args.reg)))

    # generate data matrices
    feat1 = features[get_label_indices(df=metadata, label=args.label1)]
    feat2 = features[get_label_indices(df=metadata, label=args.label2)]

    if args.split_features:
        _, labels1 = generate_feature_splits(feat1, nbins=args.nbins)
        _, labels2 = generate_feature_splits(feat2, nbins=args.nbins)

    else:
        labels1 = feat1
        labels2 = feat2

    pred = compute_pred_labels(ot_mat=ot_mat, ref_labels=labels2, num_labels=args.nbins)
    roc1 = compute_roc_score(pred_labels=pred[:,0], ref_labels=labels1==0)
    roc2 = compute_roc_score(pred_labels=pred[:,1], ref_labels=labels1==1)

    logger.info("ROC 1: %s" % str(roc1))
    logger.info("ROC 2: %s" % str(roc2))
