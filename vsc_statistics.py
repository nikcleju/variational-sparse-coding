import numpy as np
import time

class VSC_Statistics:
    def __init__(self, train_args, solver_args, encoder, decoder, patches) -> None:

        self.train_args  = train_args
        self.solver_args = solver_args

        self.encoder = encoder
        self.decoder = decoder
        self.patches = patches

        # Initialize empty arrays for tracking learning data
        #dictionary_saved = np.zeros((train_args.epochs, *dictionary.shape))
        #dictionary_saved_arranged = np.zeros((train_args.epochs, *dictionary.shape))  # For nicer plots
        self.dictionary_saved = np.zeros((train_args.epochs, *self.decoder.shape))
        self.dictionary_saved_linear_arranged = np.zeros((train_args.epochs, train_args.dict_size, self.decoder.shape[2]*self.decoder.shape[3]))  # For nicer plots
        self.dictionary_use = np.zeros((train_args.epochs, train_args.dict_size))
        self.lambda_list = np.zeros((train_args.epochs, train_args.dict_size))
        self.coeff_true = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
        #coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size))
        self.coeff_est = np.zeros((train_args.epochs, train_args.batch_size, train_args.dict_size, train_args.patch_size, train_args.patch_size))
        self.train_loss = np.zeros(train_args.epochs)
        self.val_true_recon, self.val_recon = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
        self.val_true_l1, self.val_l1 = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
        self.val_iwae_loss, self.val_kl_loss = np.zeros(train_args.epochs), np.zeros(train_args.epochs)
        self.train_time = np.zeros(train_args.epochs)

    def collect_epoch(self, epoch, epoch_loss, init_time):

        # Save and print data from epoch
        self.train_time[epoch] = time.time() - init_time
        epoch_time = self.train_time[0] if epoch == 0 else self.train_time[epoch] - self.train_time[epoch - 1]
        self.train_loss[epoch]     = np.sum(epoch_loss)       / len(train_patches)
        self.val_recon[epoch]      = np.sum(epoch_val_recon)  / len(val_patches)
        val_true_recon[epoch] = np.sum(epoch_true_recon) / len(val_patches)
        val_l1[epoch]         = np.sum(epoch_val_l1)  / len(val_patches)
        val_true_l1[epoch]    = np.sum(epoch_true_l1) / len(val_patches)
        val_iwae_loss[epoch]  = np.mean(epoch_iwae_loss)
        val_kl_loss[epoch]    = np.mean(epoch_kl_loss)
        # coeff_est[j], coeff_true[j] = b_hat.T, b_true.T  # We don't have a b_true
        coeff_est[epoch] = b_hat.detach().cpu().numpy()

        if solver_args.threshold and solver_args.solver == "VI":
            #lambda_list[j] = encoder.lambda_.data.mean(dim=(0, 1)).cpu().numpy()
            # e.g. encoder.lambda_.shape = [100, 5, 6, 16, 16]
            lambda_list[epoch] = encoder.lambda_.data.mean(dim=(0, 1, 3, 4)).cpu().numpy()  # TODO: check what should remain
        else:
            lambda_list[epoch] = np.ones(train_args.dict_size) * -1
        #dictionary_saved[j] = dictionary
        dictionary_saved[epoch] = dictionary.depthconv.weight.detach().cpu().numpy()
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
        dictionary_saved_linear_arranged[epoch] = linearize_convdict(dictionary_saved[epoch])

        if solver_args.debug:
            #print_debug(train_args, b_true.T, b_hat.T)
            print_debug(train_args, b_hat.detach().cpu().numpy(), b_hat.detach().cpu().numpy())
            logging.info("Mean lambda value: {:.6f}".format(lambda_list[epoch].mean()))
            logging.info("Mean dict norm: {}".format(np.sqrt(np.sum(linearize_convdict(dictionary_saved[epoch]) ** 2, axis=0)).mean()))
            logging.info("Est validation IWAE loss: {:.6f}".format(val_iwae_loss[epoch]))
            logging.info("Est validation KL loss: {:.6f}".format(val_kl_loss[epoch]))
            logging.info("Est validation recon loss: {:.6f}".format(val_recon[epoch]))
            logging.info("Est validation l1 loss: {:.6f}".format(val_l1[epoch]))
            logging.info("Est validation total loss = val_recon_loss + solver_args.lambda_ * val_l1_loss: {:.6f}".format(
                val_recon[epoch] + solver_args.lambda_ * val_l1[epoch])
            )
            logging.info("FISTA total loss: {:.3E}".format(val_true_recon[epoch] + solver_args.lambda_ * val_true_l1[epoch]))

    def collect_train_all():
        pass


    def collect_val_epoch():
        pass

    def collect_val_all():
        pass


class StatisticsCollector:

    def __init__(self, file_name, file_mode):
        self._file_name = file_name
        self._file_mode = file_mode

    def __enter__(self):
        self._file = open(self._file_name, self._file_mode)
        return self._file


    def __exit__(self, exc_type,exc_value, exc_traceback):
        self._file.close()
