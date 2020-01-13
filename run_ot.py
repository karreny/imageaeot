from optimal_transport.ot import get_ot_matrix
from utils import setup_logger

import argparse
import os
import sys

def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--metafile', action="store", default = "splits/train_NIH3T3.csv")
    options.add_argument('--featfile', action="store", default = "results/features/NIH3T3_features/AE_features.txt")
    options.add_argument('--save-dir', action="store", dest="save_dir", default="results/ot/test/")

    options.add_argument('--label1', action="store", default=0, type=float)
    options.add_argument('--label2', action="store", default=1, type=float)
    options.add_argument('--reg', action="store", default=1, type=float)

    return options.parse_args()

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='ot_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    get_ot_matrix(args, logger)
