from optimal_transport import get_ot_matrix, eval_ot_matrix
from utils import setup_logger

import argparse
import os
import sys

def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--metafile', action="store", default = "splits/train_NIH3T3.csv")
    options.add_argument('--featfile', action="store", default = "results/features/NIH3T3_features/AE_features.txt")
    options.add_argument('--evalfeatfile', action="store", default = "results/features/NIH3T3_features/mahotas_features_eccentriciy.txt")
    options.add_argument('--save-dir', action="store", dest="save_dir", default="results/ot/test/")
    options.add_argument('--eval-save-dir', action="store", dest="eval_save_dir", default=None)

    options.add_argument('--label1', action="store", default=0, type=float)
    options.add_argument('--label2', action="store", default=1, type=float)
    options.add_argument('--reg', action="store", default=1, type=float)
    options.add_argument('--nbins', action="store", default=2, type=int)
    options.add_argument('--split-features', action="store_true", default=False, type=bool)

    return options.parse_args()

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='ot_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    get_ot_matrix(args, logger)
    eval_ot_matrix(args, logger)
