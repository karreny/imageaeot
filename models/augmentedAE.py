import torch
import torch.nn as nn
import torch.nn.functional as F

from .AE import AE

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae

class AugmentedAE(AE):
    def __init__(self, nc=1, ngf=64, ndf=64, latent_variable_size=128, n_labels=3):
        super(AugmentedAE, self).__init__(nc=nc, ngf=ngf, ndf=ndf, latent_variable_size=latent_variable_size)

        self.n_labels = n_labels
        self.loss_names = ['recon_loss', 'class_loss', 'loss']
        self.classifier = nn.Linear(latent_variable_size, n_labels)

    def classify(self, x):
        z = self.encode(x)
        return self.classifier(z)

    def forward(self, x):
        y = x['targets']
        weights = x['weights']
        x = x['image']
        
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            y = y.cuda()
            weights = weights.cuda()

        z = self.encode(x)
        pred = self.classifier(z)
        recon = self.decode(z)
        return {'input': x, 'latent': z, 'recon': recon, 'pred': pred, 'targets': y, 'weights': weights}

    def compute_loss(self, outputs, loss_trackers=None):
        loss = self.mseloss(outputs['input'], outputs['recon'])
        class_loss = F.binary_cross_entropy_with_logits(outputs['pred'], outputs['targets'],
                                                  outputs['weights'],
                                                  pos_weight=None,
                                                  reduction='mean')
        
        if loss_trackers:
            loss_trackers['loss'].add(loss.item()+class_loss.item(), len(outputs['input']))
            loss_trackers['recon_loss'].add(loss.item(), len(outputs['input']))
            sum_weights = outputs['weights'].sum().item()/self.n_labels
            loss_trackers['class_loss'].add(class_loss.item()*len(outputs['input'])/sum_weights, sum_weights)

        return loss + class_loss

