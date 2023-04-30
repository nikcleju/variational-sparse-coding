"""
Train sparse dictionary model (Olshausen 1997) with whitened images used in the original paper. This script
applies a variational posterior to learn the sparse codes.

@Filename    train_vsc_dict.py
@Author      Kion
@Created     11/01/21
"""
import argparse
import datetime
import time
import os
import logging
import json, codecs
import itertools
from types import SimpleNamespace

import numpy as np
import torch

from compute_vsc_statistics import compute_statistics, compute_statistics_conv
from utils.dict_plotting import save_dict_fast, arrange_dict_similar
from utils.solvers import FISTA, ADMM
from model.vi_encoder import VIEncoder, VIEncoder2D
from model.util import estimate_rejection_stat
from model.scheduler import CycleScheduler
from utils.data_loader import load_whitened_images
from utils.util import *

from model.feature_enc import ConvDictDecoder

from torch.profiler import profile, record_function, ProfilerActivity

from torch.utils.tensorboard import SummaryWriter
import torchvision


if __name__ == "__main__":

    #=================
    # Arguments
    #=================
    parser = argparse.ArgumentParser(description='Variational Sparse Coding')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to config file for training.')
    args = parser.parse_args()

    with open(args.config) as json_data:
        config_data = json.load(json_data)
    train_args  = SimpleNamespace(**config_data['train'])
    solver_args = SimpleNamespace(**config_data['solver'])

    #=================
    # Debug and profiling options
    #=================

    do_profile = False
    torch.autograd.set_detect_anomaly(True, check_nan=True)

    if do_profile:
        profile_scheduler = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
        train_args.epochs = 4

    # Tensorboard stuff
    do_tensorboard = False
    tb = SummaryWriter()

    # Add default for new parameters, if not specified
    params_add_defaults(train_args, solver_args)

    #=================
    # Prepare
    #=================
    # Make folders
    os.makedirs(train_args.save_path, exist_ok=True)
    print("Save folder {}".format(train_args.save_path))

    # Save config for reference
    with open(train_args.save_path + '/config.json', 'wb') as f:
        json.dump(config_data, codecs.getwriter('utf-8')(f), ensure_ascii=False, indent=2)

    # Setup logging
    logging.basicConfig(filename=os.path.join(train_args.save_path, 'training.log'),
                        filemode='w', level=logging.DEBUG)


    #=================
    # Initialization
    #=================
    default_device = torch.device('cuda', train_args.device)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    #~~~~~~~~~~~~~~~~~~~
    # Initialize decoder
    #~~~~~~~~~~~~~~~~~~~
    if train_args.fixed_dict:
        dictionary = np.load("data/ground_truth_dict.npy")
        step_size = 0.
    else:
        # dictionary = np.random.randn(train_args.patch_size ** 2, train_args.dict_size)
        # dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))

        # Depthwise convolution:
        # TODO: Initialize explicitly?
        dictionary = ConvDictDecoder(train_args.dict_size, kernel_size=(train_args.kernel_size,train_args.kernel_size), stride=(1,1), padding='same')
        step_size = train_args.lr

        # Save initial state
        # Nic: save decoder as well
        torch.save({'model_state': dictionary.state_dict()},
                   train_args.save_path + f"dictionarystate_epoch0.pt")
        torch.save(dictionary, train_args.save_path + f"dictionary_epoch0.pt")

    #~~~~~~~~~~~~~~~~~~~
    # Load data
    #~~~~~~~~~~~~~~~~~~~
    # Use Pytorch's Dataset classes
    train_patches, val_patches = load_whitened_images(train_args, dictionary)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_patches))
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_patches))
    val_loader  = torch.utils.data.DataLoader(val_dataset,   batch_size=train_args.batch_size, shuffle=False)

    #~~~~~~~~~~~~~~~~~~~
    # Initialize encoder
    #~~~~~~~~~~~~~~~~~~~
    if solver_args.solver == "VI":

        # encoder = VIEncoder(train_args.patch_size**2, train_args.dict_size, solver_args).to(default_device)
        # vi_opt = torch.optim.SGD(itertools.chain(encoder.parameters(), dictionary.parameters()),
                                #  lr=solver_args.vsc_lr, #weight_decay=1e-4,
                                #  momentum=0.9, nesterov=True)

        encoder = VIEncoder2D(512, train_args.dict_size, solver_args, input_size=(1,512,512),
                              conv_sizes=(4, 8, 16, 32) ).to(default_device)

        # Save initial state
        if solver_args.solver == "VI":
            torch.save({'model_state': encoder.state_dict()},
                       train_args.save_path + f"encoderstate_epoch0.pt")
            torch.save(encoder, train_args.save_path + f"encoder_epoch0.pt")

        # Optimizer for the encoder
        vi_opt = torch.optim.SGD(encoder.parameters(),
                                 lr=solver_args.vsc_lr, #weight_decay=1e-4,
                                 momentum=0.9, nesterov=True)

        # Another optimizer for the dictionary
        dict_opt = torch.optim.SGD(dictionary.parameters(),
                                   lr=train_args.lr,
                                   weight_decay=1e-4,
                                   momentum=0.9, nesterov=True)

        # Scheduler
        vi_scheduler = CycleScheduler(vi_opt, solver_args.vsc_lr,
                                        n_iter=(train_args.epochs * train_patches.shape[0]) // train_args.batch_size,
                                        momentum=None, warmup_proportion=0.05)

        # Create core-set for prior
        if solver_args.prior_method == "coreset":
            build_coreset(solver_args, encoder, train_patches, default_device)
        if solver_args.sample_method == "rejection":
            encoder.rejection_stat = np.zeros(len(train_patches))

    elif solver_args.solver == "FISTA" or solver_args.solver == "ADMM":
        lambda_warmup = 0.1

    # Initialize empty arrays for tracking learning data
    #dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
    #dictionary_saved_arranged = np.zeros((train_args.epochs, *dictionary.shape))  # For nicer plots
    dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
    dictionary_saved_linear_arranged = np.zeros((train_args.epochs, train_args.dict_size, dictionary.shape[2]*dictionary.shape[3]))  # For nicer plots
    dictionary_use = np.zeros((train_args.epochs, train_args.dict_size))
    lambda_list = np.zeros((train_args.epochs, train_args.dict_size))
    coeff_true = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    #coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
    coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size, train_args.patch_size, train_args.patch_size))
    train_loss = np.zeros(train_args.epochs)
    val_true_recon, val_recon = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_true_l1, val_l1 = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    val_iwae_loss, val_kl_loss = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
    train_time = np.zeros(train_args.epochs)

    # Start a profiler, if desired
    if do_profile:
        prof = torch.profiler.profile(
            schedule=profile_scheduler,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/myprofile'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
            )
        prof.start()

    #~~~~~~~~~~~~~~~~~~~
    # Initialize encoder
    #~~~~~~~~~~~~~~~~~~~
    init_time = time.time()
    for j in range(train_args.epochs):
        if solver_args.sample_method == "rejection" and j % 20 == 0:
            encoder.rejection_stat = estimate_rejection_stat(encoder, train_patches, dictionary, train_args,
                                                             solver_args, default_device)

        epoch_loss = np.zeros(train_patches.shape[0] // train_args.batch_size)
        # Shuffle training data-set
        for i, batch in enumerate(train_loader):

            # Do not linearize
            #patches = batch[0].reshape(train_args.batch_size, -1).T
            patches = batch[0]
            patch_idx = None

            # ATTENTION:  ??
            # Full images are 1 x 512x512
            # Patches are 100 x 16 x 16
            if len(patches.shape) == 3:
                patches = torch.unsqueeze(patches, 1)  # Add channels dimension: batchsize, channels, dim1, dim2

            patches_cu = patches.float().to(default_device)

            # If requires_grad is False, the dictionary will NOT be updated until the end of an epoch
            #dict_cu = torch.tensor(dictionary, device=default_device, requires_grad=train_args.update_dict_every_step).float()
            dict_cu = dictionary.float().to(default_device)

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b = FISTA(dictionary, patches, tau=solver_args.lambda_*lambda_warmup)
                b_select = np.array(b)
                b = torch.tensor(b, device=default_device).unsqueeze(dim=0).float()
                weight = torch.ones((len(b), 1), device=default_device)
                lambda_warmup += 1e-4
                if lambda_warmup >= 1.0:
                    lambda_warmup = 1.0
            elif solver_args.solver == "ADMM":
                b = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI":
                iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, dict_cu, patch_idx)

                # Tensorboard
                if do_tensorboard:
                    if j == 0 and i == 0:
                        grid = torchvision.utils.make_grid(patches)
                        tb.add_image("images", grid)
                        #tb.add_graph(encoder, (patches_cu, dict_cu, patch_idx))
                        tb.add_graph(dict_cu, b_cu)

                # Dictionary is NOT updated when dict_cu `requires_grad` is False (torch.tensor())
                # It is now!!!
                vi_opt.zero_grad()
                dict_opt.zero_grad()

                iwae_loss.backward()

                vi_opt.step()
                dict_opt.step()
                vi_scheduler.step()

                if solver_args.true_coeff and not train_args.fixed_dict:
                    b = FISTA(dictionary, patches, tau=solver_args.lambda_)
                else:
                    # Select a single spare code out of the J ones, according to weight
                    # In their sampling method, only one is selected, use that
                    sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                    #b_select = b_cu[torch.arange(len(b_cu)), sample_idx].detach().cpu().numpy().T
                    #b_select = b_cu[torch.arange(len(b_cu)), sample_idx, :].detach().cpu().numpy()
                    b_select = b_cu[torch.arange(len(b_cu)), sample_idx, :].detach()  # Keep un CUDA
                    weight = weight.detach()
                    #b = b_cu.permute(1, 2, 0).detach()
                    #b = b_cu.detach()  # Still J samples!!
                    b = b_select

            # Take gradient step on dictionaries
            #generated_patch = dict_cu @ b
            #generated_patch = dict_cu(torcb.float().to(default_device))
            generated_patch = dict_cu(b).detach().cpu().numpy()
            # We don't need to update manually here
            # #residual = patches_cu.T - generated_patch
            # residual = patches_cu - generated_patch
            # #select_penalty = np.sqrt(np.sum(dictionary ** 2, axis=0)) > 1.5
            # residual[:, :, None].shape
            # torch.Size([20, 256, 1, 100])
            # b[:,None].shape
            # torch.Size([20, 1, 256, 100])
            # (residual[:, :, None] * b[:, None]).shape
            # torch.Size([20, 256, 256, 100])
            # weight[:,None, None].shape
            # torch.Size([100, 1, 1, 20])
            # ((residual[:, :, None] * b[:, None]) * weight.T[:, None, None]).shape
            # torch.Size([20, 256, 256, 100])
            # step = ((residual[:, :, None] * b[:, None]) * weight.T[:, None, None]).sum(axis=(0, 3)) / train_args.batch_size
            # step = step.detach().cpu().numpy() -  2*train_args.fnorm_reg*dictionary#*select_penalty
            # dictionary += step_size * step

            # Normalize dictionaries. Required to prevent unbounded growth, Tikhonov regularisation also possible.
            if train_args.normalize:
                dictionary /= np.sqrt(np.sum(dictionary ** 2, axis=0))

            # Calculate loss after gradient step
            #epoch_loss[i] = 0.5 * np.sum((patches.numpy() - dictionary @ b_select) ** 2) + solver_args.lambda_ * np.sum(np.abs(b_select))
            epoch_loss[i] = 0.5 * np.sum((patches.numpy() - generated_patch) ** 2) + solver_args.lambda_ * np.sum(np.abs(b_select.detach().cpu().numpy()))
            # Log which dictionary entries are used
            # dict_use = np.count_nonzero(b_select, axis=1)
            # dictionary_use[j] += dict_use / ((train_patches.shape[0] // train_args.batch_size))

            # Ramp up sigmoid for spike-slab
            if solver_args.prior_distribution == "concreteslab":
                encoder.temp *= 0.9995
                if encoder.temp <= solver_args.temp_min:
                    encoder.temp = solver_args.temp_min
            if solver_args.prior_method == "clf":
                encoder.clf_temp *= 0.9995
                if encoder.clf_temp <= solver_args.clf_temp_min:
                    encoder.clf_temp = solver_args.clf_temp_min
            if solver_args.prior_distribution == "concreteslab" or solver_args.prior_distribution == "laplacian":
                if ((train_patches.shape[0] // train_args.batch_size)*j + i) >= 1500:
                    encoder.warmup += 2e-4
                    if encoder.warmup >= 1.0:
                        encoder.warmup = 1.0

        # Test reconstructed or uncompressed dictionary on validation data-set
        epoch_true_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_recon = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_true_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_val_l1 = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_iwae_loss = np.zeros(val_patches.shape[0] // train_args.batch_size)
        epoch_kl_loss = np.zeros(val_patches.shape[0] // train_args.batch_size)

        for i, batch in enumerate(val_loader):
            # Load next batch of validation patches
            #patches = batch[0].reshape(train_args.batch_size, -1).T
            patches = batch[0]
            patches_idx = None

            if len(patches.shape) == 3:
                patches = torch.unsqueeze(patches, 1)

            # Infer coefficients
            if solver_args.solver == "FISTA":
                b_hat = FISTA(dictionary, patches, tau=solver_args.lambda_)
                b_true = np.array(b_hat)
                iwae_loss, kl_loss = 0., np.array(0.)
            elif solver_args.solver == "ADMM":
                b_hat = ADMM(dictionary, patches, tau=solver_args.lambda_)
            elif solver_args.solver == "VI":
                with torch.no_grad():
                    # Run VI and get sparse codes
                    # patches_cu = patches.T.float().to(default_device)
                    # dict_cu = torch.tensor(dictionary, device=default_device).float()
                    # iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, dict_cu, patches_idx)
                    # sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()  # Why?
                    # b_select = b_cu[torch.arange(len(b_cu)), sample_idx]
                    # b_hat = b_select.detach().cpu().numpy().T
                    # b_true = FISTA(dictionary, patches.numpy(), tau=solver_args.lambda_)
                    patches_cu = patches.float().to(default_device)
                    dict_cu = dictionary.float().to(default_device)
                    iwae_loss, recon_loss, kl_loss, b_cu, weight = encoder(patches_cu, dict_cu, patch_idx)
                    sample_idx = torch.distributions.categorical.Categorical(weight).sample().detach()
                    b_select = b_cu[torch.arange(len(b_cu)), sample_idx, :].detach()  # Keep on CUDA
                    weight = weight.detach()
                    b_hat = b_select

            # Compute and save loss
            # epoch_true_recon[i] = 0.5 * np.sum((patches.numpy() - dictionary @ b_true) ** 2)
            # epoch_val_recon[i] = 0.5 * np.sum((patches.numpy() - dictionary @ b_hat) ** 2)
            # epoch_true_recon[i] = 0.5 * np.sum((patches.numpy() - dict_cu(b_true) ** 2)  # We don't have a b_true!
            epoch_val_recon[i] = 0.5 * np.sum((patches.numpy() - (dict_cu(b_hat)).detach().cpu().numpy()) ** 2)
            #epoch_true_l1[i] = np.sum(np.abs(b_true))
            epoch_val_l1[i] = np.sum(np.abs(b_hat.detach().cpu().numpy()))
            epoch_iwae_loss[i] = iwae_loss
            epoch_kl_loss[i]   = kl_loss.mean()

        # Decay step-size
        step_size = step_size * train_args.lr_decay

        # Save and print data from epoch
        train_time[j] = time.time() - init_time
        epoch_time = train_time[0] if j == 0 else train_time[j] - train_time[j - 1]
        train_loss[j]     = np.sum(epoch_loss)       / len(train_patches)
        val_recon[j]      = np.sum(epoch_val_recon)  / len(val_patches)
        val_true_recon[j] = np.sum(epoch_true_recon) / len(val_patches)
        val_l1[j]         = np.sum(epoch_val_l1)  / len(val_patches)
        val_true_l1[j]    = np.sum(epoch_true_l1) / len(val_patches)
        val_iwae_loss[j]  = np.mean(epoch_iwae_loss)
        val_kl_loss[j]    = np.mean(epoch_kl_loss)
        # coeff_est[j], coeff_true[j] = b_hat.T, b_true.T  # We don't have a b_true
        coeff_est[j] = b_hat.detach().cpu().numpy()

        if solver_args.threshold and solver_args.solver == "VI":
            #lambda_list[j] = encoder.lambda_.data.mean(dim=(0, 1)).cpu().numpy()
            # e.g. encoder.lambda_.shape = [100, 5, 6, 16, 16]
            lambda_list[j] = encoder.lambda_.data.mean(dim=(0, 1, 3, 4)).cpu().numpy()  # TODO: check what should remain
        else:
            lambda_list[j] = np.ones(train_args.dict_size) * -1
        #dictionary_saved[j] = dictionary
        dictionary_saved[j] = dictionary.depthconv.weight.detach().cpu().numpy()
        # Arrange dictionary elements in a more consistent way, from one saved image to the next
        # if j != 0:
        #     dictionary_saved_arranged[j] = arrange_dict_similar(dictionary_saved[j], dictionary_saved_arranged[j-1])
        # else:
        #     dictionary_saved_arranged[j] = dictionary_saved[j]
        def linearize_convdict(dict):
            shape = dict.shape
            return dict.reshape(shape[0],-1)
        # if j != 0:
        #     dictionary_saved_linear_arranged[j] = arrange_dict_similar(linearize_convdict(dictionary_saved[j]), dictionary_saved_linear_arranged[j-1])
        # else:
        #     dictionary_saved_linear_arranged[j] = linearize_convdict(dictionary_saved[j])
        dictionary_saved_linear_arranged[j] = linearize_convdict(dictionary_saved[j])

        if solver_args.debug:
            #print_debug(train_args, b_true.T, b_hat.T)
            print_debug(train_args, b_hat.detach().cpu().numpy(), b_hat.detach().cpu().numpy())
            logging.info("Mean lambda value: {:.6f}".format(lambda_list[j].mean()))
            logging.info("Mean dict norm: {}".format(np.sqrt(np.sum(linearize_convdict(dictionary_saved[j]) ** 2, axis=0)).mean()))
            logging.info("Est validation IWAE loss: {:.6f}".format(val_iwae_loss[j]))
            logging.info("Est validation KL loss: {:.6f}".format(val_kl_loss[j]))
            logging.info("Est validation recon loss: {:.6f}".format(val_recon[j]))
            logging.info("Est validation l1 loss: {:.6f}".format(val_l1[j]))
            logging.info("Est validation total loss = val_recon_loss + solver_args.lambda_ * val_l1_loss: {:.6f}".format(
                val_recon[j] + solver_args.lambda_ * val_l1[j])
            )
            logging.info("FISTA total loss: {:.3E}".format(val_true_recon[j] + solver_args.lambda_ * val_true_l1[j]))

        if j < 10 or (j + 1) % train_args.save_freq == 0 or (j + 1) == train_args.epochs:
            # save_dict_fast(dictionary, train_args.save_path + f"dictionary_epoch{j+1}.png")
            save_dict_fast(dictionary_saved_linear_arranged[j].T, train_args.save_path + f"dict_epoch{j+1}.png", scale_factor=20)

            np.savez_compressed(train_args.save_path + f"train_savefile.npz",
                    phi=dictionary_saved, lambda_list=lambda_list, time=train_time,
                    train=train_loss, val_true_recon=val_true_recon, val_recon=val_recon,
                    val_l1=val_l1, val_true_l1=val_true_l1, val_iwae_loss=val_iwae_loss,
                    val_kl_loss=val_kl_loss, coeff_est=coeff_est, coeff_true=coeff_true,
                    dictionary_use=dictionary_use)
            if solver_args.solver == "VI":
                # Save only model parameters (state_dict)
                torch.save({'model_state': encoder.state_dict()}, train_args.save_path + f"encoderstate_epoch{j+1}.pt")
                # Also save full model
                torch.save(encoder, train_args.save_path + f"encoder_epoch{j+1}.pt")

            # Nic: save decoder as well
            torch.save({'model_state': dictionary.state_dict()}, train_args.save_path + f"dictionarystate_epoch{j+1}.pt")
            torch.save(dictionary, train_args.save_path + f"dictionary_epoch{j+1}.pt")

        logging.info("Epoch {} of {}, Avg Train Loss = {:.4f}, Avg Val Loss = {:.4f}, Time = {:.0f} secs".format(j + 1,
                                                                                                          train_args.epochs,
                                                                                                          train_loss[j],
                                                                                                          val_recon[j] + solver_args.lambda_ * val_l1[j],
                                                                                                          epoch_time))
        logging.info("\n")

        if do_profile:
            prof.step()

    if do_profile:
        prof.stop()

    if train_args.compute_stats:
        #compute_statistics(train_args.save_path, train_args, solver_args)
        compute_statistics_conv(train_args.save_path, train_args, solver_args)

    # Close tensorboard session
    if do_tensorboard:
        tb.close()
