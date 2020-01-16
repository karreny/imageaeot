import torch
from torch.utils.data import DataLoader

#from models.VAE import VAE
from models.AE import AE
from dataset.dataset import CellImageDataset
from features import extract_AE_features, extract_PCA_features, extract_mahotas_features, extract_flattened_features
from utils import setup_logger

import numpy as np
import argparse
import os
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--datadir', action="store", default = "data")
    options.add_argument('--metafile', action="store", default = "splits/train_NIH3T3.csv")
    
    options.add_argument('--save-dir', action="store", dest="save_dir", default="results/features/test/")
    options.add_argument('--pretrained-file', action="store", dest="pretrained_file", default="pretrained/NIH3T3_128_0.00000001_1950.pth")
    options.add_argument('--batch-size', action="store", dest="batch_size", default=128, type = int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)

    options.add_argument('--ae-features', action="store_true")
    options.add_argument('--pca-features', action="store_true")
    options.add_argument('--flat-features', action="store_true")
    options.add_argument('--mahotas-features', action="store_true")

    return options.parse_args()

def main(args, logger):
    
    # load data
    dataset = CellImageDataset(datadir=args.datadir, metafile=args.metafile, mode='val')
    logger.info("Dataset length is %s" % len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # load model
    #net = VAE(latent_variable_size=args.latent_dims)
    net = AE(latent_variable_size=args.latent_dims)
    net.load_state_dict(torch.load(args.pretrained_file)['state_dict'])

    logger.info(net)

    # extract AE features
    if args.ae_features:
        extract_AE_features(dataloader=dataloader, net=net, savefile=os.path.join(args.save_dir, 'AE_features.txt'))

    # extract PCA features
    if args.pca_features:
        extract_PCA_features(dataloader=dataloader, net=None, savefile=os.path.join(args.save_dir, 'PCA_features.txt'))
    
    # flatten images as features
    if args.flat_features:
        extract_flattened_features(dataloader=dataloader, net=None, savefile=os.path.join(args.save_dir, 'flattened_features.txt'))

    # extract mahotas features
    if args.mahotas_features:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        extract_mahotas_features(dataloader=dataloader, net=None, savefile=os.path.join(args.save_dir, 'mahotas_features'))

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='training_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    main(args, logger)
