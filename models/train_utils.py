import torch
import torch.optim as optim
from torchnet.meter.meter import Meter

import numpy as np

class AverageMeter(Meter):
    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def add(self, value, n):
        self.sum += value * n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.mean_old = self.mean
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n + n)
            self.mean_old = self.mean

        self.n += n

    def value(self):
        return self.mean

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = np.nan
        self.mean_old = 0.0

def train_model(trainloader, model, optimizer, target_kw='label', single_batch=False):
    '''Method for training model (updating model params) based on given criterion'''
    
    model.train()

    loss_trackers = {k: AverageMeter() for k in model.loss_names}

    for sample in trainloader:
        model.zero_grad()
        output = model(sample)

        loss = model.compute_loss(output, loss_trackers=loss_trackers)

        loss.backward()
        optimizer.step()

        if single_batch:
            break

    return {'train_'+k: loss_trackers[k].value() for k in model.loss_names}

def evaluate_model(testloader, model, target_kw='label', single_batch=False):
    '''Method for evaluating model based on given criterion'''
    
    model.eval()

    loss_trackers = {k: AverageMeter() for k in model.loss_names}

    for sample in testloader:
        with torch.no_grad():
            output = model(sample)
            _ = model.compute_loss(output, loss_trackers=loss_trackers)

        if single_batch:
            break

    return {'test_'+k: loss_trackers[k].value() for k in model.loss_names}

def save_checkpoint(current_state, filename):
    torch.save(current_state, filename)

def setup_optimizer(name, param_list):
    if name == 'sgd':
        return optim.SGD(param_list, momentum=0.9)
    elif name == 'adam':
        return optim.Adam(param_list)
    else:
        raise KeyError("%s is not a valid optimizer (must be one of ['sgd', adam']" % name)
