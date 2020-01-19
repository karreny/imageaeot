import torch
from torch.utils.data import DataLoader

from models.AE import AE
from dataset.dataset import CellImageDataset
from features import visualize_AE_recon, visualize_PCA_recon
from utils import setup_logger

import numpy as np
import argparse
import os
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--datadir', action="store", default = "data")
    options.add_argument('--metafile', action="store", default = "splits/train_total.csv")
    
    options.add_argument('--save-dir', action="store", dest="save_dir", default="results/recon/test/")
    options.add_argument('--pretrained-file', action="store", dest="pretrained_file", default="results/AE/exp1/models/best.pth")
    options.add_argument('--batch-size', action="store", dest="batch_size", default=128, type = int)
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)

    options.add_argument('--ae-features', action="store_true")
    options.add_argument('--pca-features', action="store_true")

    return options.parse_args()

def main(args, logger):
    
    # load data
    dataset = CellImageDataset(datadir=args.datadir, metafile=args.metafile, mode='val')
    logger.info("Dataset length is %s" % len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    dataloader_bs1 = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # load model
    #net = VAE(latent_variable_size=args.latent_dims)
    net = AE(latent_variable_size=args.latent_dims)
    net.load_state_dict(torch.load(args.pretrained_file)['state_dict'])

    logger.info(net)

    # extract AE features
    if args.ae_features:
        os.makedirs(os.path.join(args.save_dir, 'AE_recon'), exist_ok=True)
        visualize_AE_recon(dataloader=dataloader_bs1, net=net, savedir=os.path.join(args.save_dir, 'AE_recon'))

    # extract PCA features
    if args.pca_features:
        os.makedirs(os.path.join(args.save_dir, 'PCA_recon'), exist_ok=True)
        visualize_PCA_recon(dataloader=dataloader, dataloader_bs1=dataloader_bs1, savedir=os.path.join(args.save_dir, 'PCA_recon'),
                            n_components=args.latent_dims)
    

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='recon_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    main(args, logger)
