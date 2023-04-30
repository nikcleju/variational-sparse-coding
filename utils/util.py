import datetime
import logging
import os

import numpy as np
import torch

from sklearn_extra.cluster import KMedoids


def print_debug(train_args, b_true, b_hat):
    total_nz = np.count_nonzero(b_hat, axis=1)
    total_near_nz = np.abs(b_hat) > 1e-6
    true_total_nz = np.count_nonzero(b_true, axis=1)
    mean_coeff = np.abs(b_hat[np.nonzero(b_hat)]).mean()
    true_coeff = np.abs(b_true[np.nonzero(b_hat)]).mean()

    coeff_acc = 0
    for k in range(len(b_hat)):
        true_sup = np.nonzero(b_true[k])[0]
        est_sup = np.nonzero(b_hat[k])[0]
        missed_support = np.setdiff1d(true_sup, est_sup)
        excess_support = np.setdiff1d(est_sup, true_sup)
        coeff_acc += (b_true.shape[1] - len(missed_support) - len(excess_support)) / b_true.shape[1]
    coeff_acc /= len(b_hat)


    # Convolutional:  total_near_nz.shape = (100, 6, 16, 16)
    if len(total_near_nz.shape) == 4:
        patch_axes = (-1,-2)      # last two dimensions are the patch dimensions, e.g. 16 x 16
        b_hat_axes = (-1,-2, -3)  # last three dimensions are the patch dimensions, e.g. 6 x 16 x 16

        # Support with coefficients larger than 1e-6 (e.g. sum for all 100 x 6 images and mean, compared to 16x16)
        logging.info(f"Mean est coeff not near-zero per image: {total_near_nz.sum(axis=patch_axes).mean():.6f} / {np.prod([total_near_nz.shape[i] for i in patch_axes]):.6f}")
        
        # Support with all non-zero coefficients, even those very small
        # Sum also over -3 axis, because total_nz was summed over axis=1 in the first place
        logging.info(f"Mean est coeff support per image: {total_nz.sum(axis=b_hat_axes).mean():.6f} / {np.prod([total_nz.shape[i] for i in patch_axes]):.6f}")
        
        #logging.info(f"Mean true coeff support: {true_total_nz.mean():.3f}")
        logging.info(f"Mean est coeff magnitude: {mean_coeff}")
        #logging.info(f"Mean true coeff magnitude: {true_coeff}")
        #logging.info("L1 distance with true coeff: {:.3E}".format(np.abs(b_hat - b_true).sum()))
        #logging.info("Coeff support accuracy: {:.2f}%".format(100.*coeff_acc))

    elif len(total_near_nz.shape) == 3:
        logging.info(f"Mean est coeff near-zero: {total_near_nz.sum(axis=-1).mean():.6f}")     
        logging.info(f"Mean est coeff support: {total_nz.mean():.6f}")                         
        logging.info(f"Mean true coeff support: {true_total_nz.mean():.6f}")
        logging.info(f"Mean est coeff magnitude: {mean_coeff}")
        logging.info(f"Mean true coeff magnitude: {true_coeff}")
        logging.info("L1 distance with true coeff: {:.6E}".format(np.abs(b_hat - b_true).sum()))
        logging.info("Coeff support accuracy: {:.2f}%".format(100.*coeff_acc))

def build_coreset(solver_args, encoder, train_patches, default_device):
    mbed_file = np.load(solver_args.coreset_embed_path)
    if solver_args.coreset_feat == "pca":
        feat = mbed_file['pca_mbed']
    elif solver_args.coreset_feat == "isomap":
        feat = mbed_file['isomap_mbed']
    elif solver_args.coreset_feat == "wavelet":
        feat = mbed_file['wavelet_mbed']
    else:
        raise NotImplementedError

    logging.info(f"Building core-set using {solver_args.coreset_alg} with {solver_args.coreset_size} centroids...")
    if solver_args.coreset_alg == "kmedoids":
        kmedoid = KMedoids(n_clusters=solver_args.coreset_size, random_state=0).fit(feat)                
        encoder.coreset = torch.tensor(train_patches[kmedoid.medoid_indices_], device=default_device).reshape(solver_args.coreset_size , -1)
        encoder.coreset_labels = torch.tensor(kmedoid.labels_, device=default_device)   
        encoder.coreset_coeff = torch.tensor(mbed_file['codes'][kmedoid.medoid_indices_], device=default_device)              
    else:
        raise NotImplementedError
    logging.info(f"...core-set succesfully built.")

def params_add_defaults(train_args, solver_args, date_path=True):
    """Utility function to add new parameters with default values
       for train_args and save_args, in case of newly added parameters
    """

    # Add new fields
    if not hasattr(solver_args, 'sparse_KL'):
        setattr(solver_args, 'sparse_KL', False)
    if not hasattr(solver_args, 'num_ISTA'):
        setattr(solver_args, 'num_ISTA', 0)
    if not hasattr(solver_args, 'ISTA_c_prior_size'):
        setattr(solver_args, 'ISTA_c_prior_size', 1)
    if not hasattr(train_args, 'update_dict_every_step'):
        setattr(train_args, 'update_dict_every_step', False)
    if not hasattr(train_args, 'kernel_size'):
        setattr(train_args, 'kernel_size', 5)
    if not hasattr(train_args, 'is2D'):
        setattr(train_args, 'is2D', False)        

    # Prepend current date and time to savepath
    if date_path:
        path_splits = train_args.save_path.split(os.sep)
        path_splits[-2] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_") + path_splits[-2]
        train_args.save_path = os.sep.join(path_splits)

