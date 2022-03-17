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
import numpy as np

import torch
import torch.nn.functional as F

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from dataset import augment, distrib
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from model.FullSubNet.mask import build_complex_ideal_ratio_mask, decompress_cIRM
from model.FullSubNet.feature import drop_band

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, data, model, estimator, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.estimator = estimator
        # if self.estimator is not None:
        #     self.estimator.eval()
        self.optimizer = optimizer

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
        if args.weightPath is not None:
            self.weight = torch.from_numpy(np.load(args.weightPath)).cuda()
            # self.weight = torch.nn.functional.normalize(self.weight, dim=0)
        else:
            self.weight = None
        self._reset()

    def _serialize(self):
        package = {}
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

        logger.info('Enhance and save samples...')
        out_dir = os.path.join(self.args.savePath, "0")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        enhance(self.args, self.model, out_dir)

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            if self.estimator is not None:
                self.estimator.train()
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
                if self.estimator is not None:
                    self.estimator.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
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
                with swap_state(self.model, self.best_state):
                    pesq, stoi = evaluate(self.args, self.model, self.tt_loader)

                metrics.update({'pesq': pesq, 'stoi': stoi})

                # enhance some samples
                logger.info('Enhance and save samples...')
                out_dir = os.path.join(self.args.savePath, str(epoch + 1))
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                enhance(self.args, self.model, out_dir)


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

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            data = [x.to(self.device) for x in data]
            noisy = data[0]
            clean = data[1]
            spec = data[3]
            spec = spec.transpose(1, 2)
            if not cross_valid:
                sources = torch.stack([noisy - clean, clean])
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean
            if self.args.model == "Demucs":
                estimate, _ = self.dmodel(noisy)
                # apply a loss function after each layer
                with torch.autograd.set_detect_anomaly(True):
                    if self.args.loss == 'l1':
                        loss = F.l1_loss(clean, estimate)
                    elif self.args.loss == 'l2':
                        loss = F.mse_loss(clean, estimate)
                    elif self.args.loss == 'huber':
                        loss = F.smooth_l1_loss(clean, estimate)
                    else:
                        raise ValueError(f"Invalid loss {self.args.loss}")
                    # MultiResolution STFT loss
                    if self.args.stft_loss:
                        sc_loss, mag_loss = self.mrstftloss(estimate.squeeze(1), clean.squeeze(1))
                        loss += sc_loss + mag_loss
            elif self.args.model == "FullSubNet":
                noisy_mag, noisy_phase, noisy_real, noisy_imag = self.dmodel.stft(torch.squeeze(noisy))
                _, _, clean_real, clean_imag = self.dmodel.stft(torch.squeeze(clean))
                cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
                cIRM = drop_band(
                    cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                    self.dmodel.num_groups_in_drop_band
                ).permute(0, 2, 3, 1)
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM = self.dmodel(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                with torch.autograd.set_detect_anomaly(True):
                    if self.args.loss == 'l1':
                        loss = F.l1_loss(cIRM, cRM)
                    elif self.args.loss == 'l2':
                        loss = F.mse_loss(cIRM, cRM)
                    elif self.args.loss == 'huber':
                        loss = F.smooth_l1_loss(cIRM, cRM)
                    print(loss)

            else:
                raise NotImplementedError
                
            with torch.autograd.set_detect_anomaly(True):
                if self.estimator is not None:
                    egemaps_func = data[2]
                    egemaps_lld = data[4]
                    encoded_out = self.estimator(spec).encoder_out.global_sample
                    estimated_egemaps = self.dmodel.fc(encoded_out)
                    if self.args.egemaps_type == "functionals":
                        true_egemaps = egemaps_func
                        if self.weight is not None:
                            egemaps_loss = torch.norm(self.weight[88:, -1] *estimated_egemaps + self.weight[:88, -1] * egemaps_func)
                        else:
                            egemaps_loss = F.mse_loss(estimated_egemaps, egemaps_func)

                    elif self.args.egemaps_type == "lld":
                        egemaps_loss = F.mse_loss(estimated_egemaps, egemaps_lld)
                    print("*****")
                    print(egemaps_loss)
                    print(loss)
                    loss += self.args.egemaps_factor * egemaps_loss

                # optimize model in training mode
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del loss
        return distrib.average([total_loss / (i + 1)], i + 1)[0]


