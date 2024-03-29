# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time
from tqdm import tqdm
import functools

import torch
import torch.nn.functional as F
import torchaudio
import collections

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from dataset import augment, distrib
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress

logger = logging.getLogger(__name__)


class Trainer_est(object):
    def __init__(self, data, estimator, decoder, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        if decoder is None:
            self.model = estimator
        else:
            assert args.model == 'decoder'
            self.model = decoder
            self.estimator = estimator
        self.dmodel = distrib.wrap(self.model)
        self.optimizer = optimizer
        window_fn = functools.partial(torch.hann_window, device=args.device)
        self.spectrogram = torchaudio.transforms.Spectrogram(hop_length=512, window_fn=window_fn)

        # data augment
        augments = []
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho))
        self.augment = torch.nn.Sequential(*augments)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(os.path.join(args.savePath, args.checkpoint_file))
            self.best_file = Path(os.path.join(args.savePath, args.best_file))
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = os.path.join(args.savePath, args.history_file)

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)
        self._reset()

    def _serialize(self):
        package = {}
        if isinstance(self.model, torch.nn.DataParallel):
            package['model'] = serialize_model(self.model.module)
        else:
            package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']
        continue_pretrained = self.args.continue_pretrained
        if continue_pretrained:
            raise NotImplementedError

    def train(self):
        if self.args.save_again:
            self._serialize()
            return
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                    f'Train Summary | End of Epoch {epoch + 1} | '
                    f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}')

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, label='Valid')
                logger.info(
                        f'Valid Summary | End of Epoch {epoch + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}')
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'epoch': epoch, 'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            # Save the best model
            if valid_loss == best_loss:
                logger.info('New best valid loss %.4f', valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                test_loss = self._run_one_epoch(epoch, label='Test')
                metrics.update({'test': test_loss})

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items() if k != 'epoch')
            logger.info('-' * 70)
            logger.info(f"Overall Summary | Epoch {epoch + 1} | {info}")

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())

    def _run_one_epoch(self, epoch, label="Train"):
        total_loss = 0
        if label == 'Train':
            data_loader = self.tr_loader
        elif label == 'Valid':
            data_loader = self.cv_loader
        elif label == 'Test':
            data_loader = self.tt_loader
        else:
            raise NotImplementedError

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        
        for i, data in tqdm(enumerate(logprog), total=len(data_loader)):
            data = [x.to(self.device) for x in data]
            noisy = data[0]
            clean = data[1]
            egemaps_func = data[2].squeeze(1).float()
            # egemaps_func = torch.rand(256, 88).cuda()
            if self.args.model == "VAE":
                # spec = data[4].squeeze(1)
                # spec = spec.transpose(1, 2)
                spec = self.spectrogram(clean).squeeze(dim=1).transpose(1, 2)
                spec = F.normalize(spec)
                estimate = self.dmodel(spec)
                # apply a loss function after each layer
                    # if self.args.loss == 'l1':
                    #     loss = F.l1_loss(egemaps, estimate)
                    # elif self.args.loss == 'l2':
                    #     loss = F.mse_loss(egemaps, estimate)
                    # elif self.args.loss == 'huber':
                    #     loss = F.smooth_l1_loss(egemaps, estimate)
                    # else:
                    #     raise ValueError(f"Invalid loss {self.args.loss}")
                losses = self.mi_loss(spec, estimate, beta_kl=1., beta_mi=1.)
                loss = losses.loss
            
            elif self.args.model == 'M5':
                estimate = self.dmodel(clean)
                loss = F.mse_loss(estimate, egemaps_func)

            elif self.args.model == 'decoder':
                # spec = data[4].squeeze(1)
                # spec = spec.transpose(1, 2)
                spec = self.spectrogram(clean).squeeze(dim=1).transpose(1, 2)
                spec = F.normalize(spec)
                encoded_out = self.estimator(spec).encoder_out.global_sample
                estimate = self.dmodel(encoded_out)
                loss = F.mse_loss(estimate, egemaps_func)
            else:
                raise NotImplementedError

            # optimize model in training mode
            with torch.autograd.set_detect_anomaly(True):
                if label == 'Train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del loss, estimate
        return distrib.average([total_loss / (i + 1)], i + 1)[0]


    def mi_loss(self, input, outputs, beta_kl=10., beta_mi=10.):
        reconstruction = outputs.decoder_out
        global_sample = outputs.encoder_out.global_sample
        local_sample = outputs.encoder_out.local_sample
        
        prior = torch.distributions.Normal(torch.zeros(local_sample.size()).cuda(), torch.ones(local_sample.size()).cuda())
        data_prop = torch.distributions.Normal(reconstruction, 0.01*torch.ones(reconstruction.size()).cuda())
        prior_ll = torch.mean(prior.log_prob(local_sample))

        global_sample_repeated = global_sample
        
        z_prediction_ll = torch.mean(outputs.predictor_out.log_prob(global_sample_repeated))
        
        reconstruction_ll = -(F.mse_loss(reconstruction, input, size_average=False)/(input.size(0)))/input.size(1)
        
        z_local_entropy = -torch.mean(outputs.encoder_out.local_dist.log_prob(local_sample))
        z_global_entropy = -torch.mean(outputs.encoder_out.global_dist.log_prob(global_sample))
        KL_local_prior = prior_ll + z_local_entropy
        # first term is a cross-entropy from prediction and prior, together with the entropy this is the mutual information
        MI_global_prediction = z_prediction_ll + z_global_entropy
        
        Loss = - reconstruction_ll - beta_kl * KL_local_prior - beta_mi * MI_global_prediction
        
        VAELosses = collections.namedtuple("Losses", ["loss", "reconstruction_nll", "prior_nll", "z_prediction_nll", "z_global_entropy", "z_local_entropy"])
        Losses = VAELosses(loss=Loss, reconstruction_nll=-reconstruction_ll, prior_nll=-prior_ll, z_prediction_nll=-z_prediction_ll, z_global_entropy=z_global_entropy, z_local_entropy=z_local_entropy)
        
        return Losses