import sys
import logging
sys.path.append('.')

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.distributions import gamma as gamma

from model.reparameterization import *

from model.feature_enc import MLPEncoder, ConvEncoder

class VIEncoder(nn.Module):

    def __init__(self, img_size, dict_size, solver_args, input_size=(3, 64, 64)):
        super(VIEncoder, self).__init__()
        self.solver_args = solver_args
        self.dict_size = dict_size

        if self.solver_args.feature_enc == "MLP":
            self.enc = MLPEncoder(int(np.sqrt(img_size)))
            input_size = (img_size**2)
        elif self.solver_args.feature_enc == "CONV":
            self.enc = ConvEncoder(img_size, input_size[0], im_size=input_size[1])
            if input_size[1] == 64:
                img_size = 256
            elif input_size[1] == 28:
                img_size = 128
        elif self.solver_args.feature_enc == "RES":
            self.enc = models.resnet18()
            self.enc.fc = nn.Identity()
            img_size = 512
        else:
            raise NotImplementedError

        self.scale = nn.Linear(img_size, dict_size)
        self.shift = nn.Linear(img_size, dict_size)

        # Number of ISTA unwraps
        self.num_ISTA = solver_args.num_ISTA

        if self.solver_args.prior_distribution == "concreteslab":
            self.spike = nn.Linear(img_size, dict_size)
            self.temp = 1.0
            self.warmup = 0.1
        if self.solver_args.threshold and self.solver_args.theshold_learn:
            self.lambda_prior_alpha = nn.Linear(img_size, dict_size)
            self.lambda_prior_beta = nn.Linear(img_size, dict_size)

            # Add layers for ISTA
            ISTA_lambda_prior_alpha = []
            ISTA_lambda_prior_beta = []
            for _ in range(self.num_ISTA):
                ISTA_lambda_prior_alpha.append(nn.Linear(dict_size, solver_args.ISTA_c_prior_size))
                ISTA_lambda_prior_beta.append(nn.Linear(dict_size, solver_args.ISTA_c_prior_size))

            self.ISTA_c_prior_alpha = nn.Linear(dict_size, solver_args.ISTA_c_prior_size)
            self.ISTA_c_prior_beta = nn.Linear(dict_size, solver_args.ISTA_c_prior_size)

            self.ISTA_lambda_prior_alpha = nn.ModuleList(ISTA_lambda_prior_alpha)
            self.ISTA_lambda_prior_beta  = nn.ModuleList(ISTA_lambda_prior_beta)

        if self.solver_args.prior_distribution == "laplacian":
            self.warmup = 0.1

        if self.solver_args.prior_method == "vamp" or self.solver_args.prior_method == "clf":
            pseudo_init = torch.randn(self.solver_args.num_pseudo_inputs, *input_size)
            self.pseudo_inputs = nn.Parameter(pseudo_init, requires_grad=True)
        if self.solver_args.prior_method == "clf":
            self.clf_temp = 1.0
            if self.solver_args.feature_enc == "MLP":
                self.clf = nn.Sequential(
                            MLPEncoder(int(np.sqrt(img_size))),
                            nn.Linear(img_size, self.solver_args.num_pseudo_inputs)
                        )
            else:
                self.clf = models.resnet18() 
                self.clf.fc = nn.Linear(512, self.solver_args.num_pseudo_inputs)

    def ramp_hyperparams(self):
        self.temp = 1e-2
        self.clf_temp = 1e-2
        self.warmup = 1.0

    def soft_threshold(self, z):
        return F.relu(torch.abs(z) - torch.abs(self.lambda_)) * torch.sign(z)

    # Nic
    def soft_threshold_tanh(self, z):
        """Use tanh() for approximate soft thresholding
        """
        z_out    = z - self.lambda_ * F.tanh(z/self.lambda_)
        non_zero = torch.nonzero(torch.gt(torch.abs(z), torch.abs(self.lambda_)), as_tuple=True)  # compare >
        oui_zero = torch.nonzero(torch.le(torch.abs(z), torch.abs(self.lambda_)), as_tuple=True)  # compare <=
        return z_out, non_zero, oui_zero

    def soft_threshold_tanh_shift(self, z, shift):
        """Use tanh() for approximate soft thresholding, then add shift
        """
        return (z - self.lambda_ * F.tanh(z/self.lambda_)) + shift * (F.tanh(z/self.lambda_))

    @classmethod
    def soft_threshold_lambda(cls, z, lmbd):
        """Soft thresholding with `threshold given as parameter
        """
        return F.relu(torch.abs(z) - torch.abs(lmbd)) * torch.sign(z)

    def ISTA_layer(self, z, A, y, i):
        """ Non-VAE ISTA iteration
        """

        # DRAFT! 
        # 
        # EXPERIMENTAL!

        # Note:
        # x_hat = (z @ A.T)

        # A.shape: torch.Size([256, 256])
        # y.shape: torch.Size([100, 20, 256])
        # z.shape: torch.Size([100, 20, 256])

        if (i < 3):
            c = self.c
        else:
            c = self.c.detach()

        S = self.eye - (1 / c) * A.T.mm(A)                            # Compute I - 1/L * D^T D

        #l = self.ISTA_lambda_tensor / self.c  # ?
        # Use same lambda ? One per coefficient
        l = self.lambda_.detach() / c

        return self.soft_threshold_lambda(torch.matmul(z, S) + (1 / c) * torch.matmul(y, A), l)      #   z = x = S( (I - 1/L D^T D) x + 1/L D^T y)

    def ISTA_layer_VAE(self, z, A, y, i, c=None):
        """ISTA layer with VAE 
        """

        # Note:
        # x_hat = (z @ A.T)

        c = c if c is not None else torch.abs(self.ISTA_c)
        c = 3 #!!!!
        l = torch.abs(self.ISTA_lambda_[i] / c)

        zout = self.soft_threshold_lambda(z - (1 / c) * z @ A.T @ A + (1 / c) * torch.matmul(y, A), l)      #   z = x = S( (I - 1/L D^T D) x + 1/L D^T y)
        norm1 = np.linalg.norm((y - z@A.T).cpu().detach().numpy())     # for debugging
        norm2 = np.linalg.norm((y - zout@A.T).cpu().detach().numpy())  # for debugging
        return zout


    def forward(self, x, decoder, idx=None):

        # Feature extraction 
        feat = self.enc(x)

        # Parameters for base distribution of sparse coefficients
        b_logscale = self.scale(feat)
        b_shift = self.shift(feat)

        if self.solver_args.threshold:
            if self.solver_args.theshold_learn:

                # Parameters for the distribution of lambda (a Gamma distribution)
                alpha = self.lambda_prior_alpha(feat).exp().clip(1e-6, 1e6)
                beta = self.lambda_prior_beta(feat).exp().clip(1e-6, 1e6)
                gamma_pred = gamma.Gamma(alpha, beta)
                gamma_prior = gamma.Gamma(3, (3 * torch.ones_like(beta)) / self.solver_args.threshold_lambda)

                # Parameters for the ISTA parameter distributions
                ISTA_alpha = []
                ISTA_beta = []
                ISTA_gamma_pred = []
                ISTA_gamma_prior = []
                for i in range(self.num_ISTA):
                    ISTA_alpha.append( self.ISTA_lambda_prior_alpha[i](feat).exp().clip(1e-6, 1e6) )
                    ISTA_beta.append( self.ISTA_lambda_prior_beta[i](feat).exp().clip(1e-6, 1e6) )
                    ISTA_gamma_pred.append( gamma.Gamma(ISTA_alpha[i], ISTA_beta[i]) )
                    ISTA_gamma_prior.append( gamma.Gamma(3, (3 * torch.ones_like(ISTA_beta[i])) / self.solver_args.threshold_lambda) )
                ISTA_c_alpha = self.ISTA_c_prior_alpha(feat).exp().clip(1e-6, 1e6)
                ISTA_c_beta  = self.ISTA_c_prior_beta(feat).exp().clip(1e-6, 1e6)
                ISTA_c_pred  = gamma.Gamma(ISTA_c_alpha, ISTA_c_beta)
                ISTA_c_prior = gamma.Gamma(3, (3 * torch.ones_like(ISTA_c_beta)) / self.solver_args.threshold_lambda)

                # Sample lambdas, and compute KL divergence term for lambda
                # rasmple() makes the gradients flow to the parameters alpha and beta
                self.lambda_ = gamma_pred.rsample([self.solver_args.num_samples]).transpose(1, 0)
                self.lambda_kl_loss = torch.distributions.kl.kl_divergence(gamma_pred, gamma_prior)

                # Same for the ISTA distributions
                self.ISTA_lambda_ = []
                self.ISTA_lambda_kl_loss = []
                for i in range(self.num_ISTA):
                    self.ISTA_lambda_.append( ISTA_gamma_pred[i].rsample([self.solver_args.num_samples]).transpose(1, 0) )
                    self.ISTA_lambda_kl_loss.append( torch.distributions.kl.kl_divergence(ISTA_gamma_pred[i], ISTA_gamma_prior[i]) )
                self.ISTA_c = ISTA_c_pred.rsample([self.solver_args.num_samples]).transpose(1, 0)
                self.ISTA_c_kl_loss = torch.distributions.kl.kl_divergence(ISTA_c_pred, ISTA_c_prior)

            else:
                self.lambda_ = torch.ones_like(b_logscale) * self.solver_args.threshold_lambda
                self.lambda_ = self.lambda_.repeat(self.solver_args.num_samples, 
                                                   *torch.ones(self.lambda_.dim(), dtype=int)).transpose(1, 0)


        if self.solver_args.prior_distribution == "laplacian":
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight = sample_laplacian(b_shift, b_logscale, x, decoder,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "gaussian":
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight  = sample_gaussian(b_shift, b_logscale, x, decoder,
                                                                 self, self.solver_args, idx=idx)
        elif self.solver_args.prior_distribution == "concreteslab":
            logspike = -F.relu(-self.spike(feat))
            iwae_loss, recon_loss, kl_loss, sparse_codes, weight = sample_concreteslab(b_shift, b_logscale, logspike, x, decoder, 
                                                                    self, self.solver_args, 
                                                                    self.temp, self.solver_args.spike_prior,
                                                                    idx=idx)       
        else:
            raise NotImplementedError
        
        return iwae_loss, recon_loss, kl_loss, sparse_codes, weight