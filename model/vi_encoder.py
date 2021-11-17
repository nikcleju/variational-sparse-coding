import sys
import logging
sys.path.append('.')

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import model.lista as lista
from model.reparameterization import *
from model.feature_enc import MLPEncoder

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, solver_args):
        super(VIEncoder, self).__init__()
        self.solver_args = solver_args
        if self.solver_args.feature_enc == "MLP":
            self.enc = MLPEncoder(img_size)
        else:
            raise NotImplementedError
        self.scale = nn.Linear((img_size**2), dict_size)
        self.shift = nn.Linear((img_size**2), dict_size)
        if self.solver_args.prior == "spikeslab":
            self.spike = nn.Linear((img_size**2), dict_size)
            self.c = 50
        elif self.solver_args.prior == "concreteslab":
            self.spike = nn.Linear((img_size**2), dict_size)
            self.temp = 1.0
        elif self.solver_args.prior == "vampprior":
            pseudo_init = torch.randn(self.solver_args.num_pseudo_inputs, (img_size**2))
            self.pseudo_inputs = nn.Parameter(pseudo_init, requires_grad=True)

    def forward(self, x, A):
        feat = self.enc(x)
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.solver_args.prior == "laplacian":
            iwae_loss, recon_loss, kl_loss, b = sample_laplacian(b_shift, b_logscale, x, A,
                                                                 self.solver_args)
        elif self.solver_args.prior == "gaussian":
            iwae_loss, recon_loss, kl_loss, b  = sample_gaussian(b_shift, b_logscale, x, A,
                                                                 self.solver_args)
        elif self.solver_args.prior == "spikeslab":
            logspike = -F.relu(-self.spike(feat))
            iwae_loss, recon_loss, kl_loss, b = sample_spikeslab(b_shift, b_logscale, logspike, x, A, 
                                                                 self.solver_args, self.c, self.solver_args.spike_prior)
        elif self.solver_args.prior == "concreteslab":
            logspike = -F.relu(-self.spike(feat))
            iwae_loss, recon_loss, kl_loss, b = sample_concreteslab(b_shift, b_logscale, logspike, x, A, 
                                                                    self.solver_args, self.temp, self.solver_args.spike_prior)       
        elif self.solver_args.prior == "vampprior":
            pseudo_feat = self.enc(self.pseudo_inputs)
            pseudo_shift, pseudo_logscale = self.shift(pseudo_feat), self.scale(pseudo_feat)
            iwae_loss, recon_loss, kl_loss, b = sample_vampprior(b_shift, b_logscale, pseudo_shift, pseudo_logscale, x, A, 
                                                                    self.solver_args)
        else:
            raise NotImplementedError

        if self.solver_args.threshold:
            b = lista.soft_threshold(b, torch.tensor(self.solver_args.threshold_lambda, device=b.device))
        
        return iwae_loss, recon_loss, kl_loss, b