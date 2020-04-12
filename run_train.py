import torch
from torch.utils.data import DataLoader

from models import model_dict
from models.train_utils import train_model, evaluate_model, save_checkpoint, setup_optimizer
from dataset import dataset_dict
from utils import setup_logger

import numpy as np
import argparse
import os
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('--datadir', action="store", default="data")
    options.add_argument('--train-metafile', action="store", default="splits/train_total.csv")
    options.add_argument('--val-metafile', action="store", default="splits/val_total.csv")
    options.add_argument('--save-dir', action="store", default='results/AE/')
    options.add_argument('--save-freq', action="store", default=50, type=int)
    options.add_argument('--seed', action="store", default=42, type=int)

    # model parameters
    options.add_argument('--optimizer', action="store", dest="optimizer", default='adam')
    options.add_argument('--latent-dims', action="store", dest="latent_dims", default=128, type = int)
    options.add_argument('--model-type', action="store", dest="model_type", default='AE')

    # training parameters
    options.add_argument('--dataset-type', action="store", default='default')
    options.add_argument('--batch-size', action="store", dest="batch_size", default=128, type=int)
    options.add_argument('--num-workers', action="store", dest="num_workers", default=8, type=int)
    options.add_argument('--learning-rate', action="store", dest="learning_rate", default=1e-3, type=float)
    options.add_argument('--max-epochs', action="store", dest="max_epochs", default=2000, type=int)
    options.add_argument('--weight-decay', action="store", dest="weight_decay", default=1e-5, type=float)

    # debugging mode
    options.add_argument('--debug-mode', action="store_true", default=False)

    return options.parse_args()

def run_training(args, logger):
    
    os.makedirs(os.path.join(args.save_dir, "models"), exist_ok=True)

    # seed run
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # load data
    trainset = dataset_dict[args.dataset_type](datadir=args.datadir, metafile=args.train_metafile, mode='train')
    testset = dataset_dict[args.dataset_type](datadir=args.datadir, metafile=args.val_metafile, mode='val')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # load model
    net = model_dict[args.model_type](latent_variable_size=args.latent_dims)
    logger.info(net)

    if torch.cuda.is_available():
        net.cuda()

    # setup optimizer and scheduler
    optimizer = setup_optimizer(name=args.optimizer, param_list=[{'params': net.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}])

    # main training loop
    best_loss = np.inf

    for epoch in range(args.max_epochs):
        logger.info("Epoch %s:" % epoch)

        train_summary = train_model(trainloader=trainloader, model=net, optimizer=optimizer, single_batch=args.debug_mode)
        logger.info("Training summary: %s" % train_summary)

        test_summary = evaluate_model(testloader=testloader, model=net, single_batch=args.debug_mode)
        logger.info("Evaluation summary: %s" % test_summary)

        current_state = {'epoch': epoch, 'state_dict': net.cpu().state_dict(), 'optimizer': optimizer.state_dict()}

        if test_summary['test_loss'] < best_loss:
            best_loss = test_summary['test_loss']
            current_state['best_loss'] = best_loss
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir,"models/best.pth"))

        logger.info("Best loss: %s" % best_loss)
        
        if epoch % args.save_freq == 0:
            save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/epoch_%s.pth" % epoch))

        save_checkpoint(current_state=current_state, filename=os.path.join(args.save_dir, "models/last.pth"))

        if torch.cuda.is_available():
            net.cuda()

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name='training_log', save_dir=args.save_dir)
    logger.info(" ".join(sys.argv))
    logger.info(args)
    run_training(args, logger)
