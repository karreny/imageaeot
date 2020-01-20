import torch
import torch.nn as nn
from torch.autograd import Variable

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae

class AE(nn.Module):
    def __init__(self, nc=1, ngf=64, ndf=64, latent_variable_size=128):
        super(AE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.loss_names = ['recon_loss', 'loss']

        self.encoder = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )

        self.fc1 = nn.Linear(ndf*8*2*2, latent_variable_size)
        
        # decoder
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 64 x 64
        )

        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*2)
        self.mseloss = nn.MSELoss()

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*2*2)
        return self.fc1(h)

    def decode(self, z):
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        x = x['image']
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        z = self.encode(x)
        recon = self.decode(z)
        return {'input': x, 'latent': z, 'recon': recon}

    def compute_loss(self, outputs, loss_trackers=None):
        loss = self.mseloss(outputs['input'], outputs['recon'])

        if loss_trackers:
            loss_trackers['loss'].add(loss.item(), len(outputs['input']))
            loss_trackers['recon_loss'].add(loss.item(), len(outputs['input']))

        return loss

