#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Muqiao Yang <muqiaoy@andrew.cmu.edu>


import logging
import argparse
import json
import os
import yaml
import random
import numpy as np
from glob import glob
from pathlib import Path
import shutil
import opensmile

import torch
import torch.nn as nn

from model import *
from model.egemaps_estimator import Egemaps_estimator, SelfAttentionPooling
from model.vae import VAE
from dataset import distrib
from dataset.dataset import NoisyCleanSet
from trainer.trainer import Trainer


def main(args):
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)

    # log rotation
    logname = "%s/train.log" % args.savePath
    max_num_log_files = 100
    for i in range(max_num_log_files - 1, -1, -1):
        if i == 0:
            p = Path(logname)
            pn = p.parent / (p.stem + ".1" + p.suffix)
        else:
            _p = Path(logname)
            p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
            pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

        if p.exists():
            if i == max_num_log_files - 1:
                p.unlink()
            else:
                shutil.move(p, pn)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s (%(filename)s:%(lineno)d) %(message)s", filename=logname, filemode='w')
    logging.info(args)
    smile_F = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals)

    if args.model == 'NSNet2':
        # raise NotImplementedError
        model = NSNet2(modelfile=args.modelPath, cfg=args.cfg)
        print(model.model)
        print(type(model.model.Shape_21))
        for n, v in model.state_dict().items():
            print(n)
            print(v.shape)
        logging.info("Loaded checkpoint from %s" % args.modelPath)
    elif args.model == 'Demucs':
        model = Demucs(**args.demucs, sample_rate=args.fs)
        state_dict = torch.load(args.modelPath)
        model.load_state_dict(state_dict)

        if args.estimatorPath is not None:
            # estimator = Egemaps_estimator(smile_F=smile_F)
            estimator = VAE(**args.vae)
            package = torch.load(args.estimatorPath)
            estimator.load_state_dict(package['state'], strict=False)
            logging.info("Loaded checkpoint from %s and %s" % (args.modelPath, args.estimatorPath))
        else:
            estimator = None
            logging.info("Loaded checkpoint from %s" % (args.modelPath))

    elif args.model == 'FullSubNet':
        model = FullSubNet(**args.fullsubnet, sample_rate=args.fs)
        state_dict = torch.load(args.modelPath)["model"]
        model.load_state_dict(state_dict)
        if args.estimatorPath is not None:
            # estimator = Egemaps_estimator(smile_F=smile_F)
            estimator = VAE(**args.vae)
            package = torch.load(args.estimatorPath)
            estimator.load_state_dict(package['state'], strict=False)
            logging.info("Loaded checkpoint from %s and %s" % (args.modelPath, args.estimatorPath))
        else:
            estimator = None
            logging.info("Loaded checkpoint from %s" % (args.modelPath))
    else:
        raise NotImplementedError(args.model)
        
    length = int(args.segment * args.fs)
    stride = int(args.stride * args.fs)
    tr_noisy_dir = os.path.join(args.dataPath, args.trainPath, "noisy")
    tr_clean_dir = os.path.join(args.dataPath, args.trainPath, "clean")
    tr_dataset = NoisyCleanSet(tr_noisy_dir, tr_clean_dir, num_files=args.num_train_files, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_train_path, egemaps_lld_path=args.egemaps_lld_train_path, spec_path=args.spec_train_path)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Total number of train files: %s" % len(tr_dataset))
    cv_noisy_dir = os.path.join(args.dataPath, args.validPath, "noisy")
    cv_clean_dir = os.path.join(args.dataPath, args.validPath, "clean")
    cv_dataset = NoisyCleanSet(cv_noisy_dir, cv_clean_dir, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_valid_path, egemaps_lld_path=args.egemaps_lld_valid_path, spec_path=args.spec_valid_path)
    cv_loader = distrib.loader(
        cv_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Total number of valid files: %s" % len(cv_dataset))
    tt_noisy_dir = os.path.join(args.dataPath, args.testPath, "noisy")
    tt_clean_dir = os.path.join(args.dataPath, args.testPath, "clean")
    tt_dataset = NoisyCleanSet(tt_noisy_dir, tt_clean_dir, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_test_path, egemaps_lld_path=args.egemaps_lld_test_path, spec_path=args.spec_test_path)
    tt_loader = distrib.loader(
        tt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Total number of test files: %s" % len(tt_dataset))

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        model.cuda()
        if estimator is not None:
            if args.egemaps_type == 'lld':
                model.fc = nn.Sequential(
                        nn.Conv1d(256, 1024, kernel_size=3),
                        nn.BatchNorm1d(1024),
                        nn.ReLU(),
                        nn.Conv1d(1024, 2048, kernel_size=3),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Conv1d(2048, 2996, kernel_size=3),
                        nn.BatchNorm1d(2996),
                        nn.ReLU(),
                        nn.Linear(932, 128),
                        nn.ReLU(),
                        nn.Linear(128, 25)).cuda()
            elif args.egemaps_type == 'functionals':
                model.fc = nn.Sequential(
                        SelfAttentionPooling(256),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 88)
                        ).cuda()
            else:
                raise NotImplementedError
            estimator.cuda()
        else:
            estimator = None

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, args.beta2))
    else:
        raise NotImplementedError
        
    trainer = Trainer(data, model, estimator, optimizer, args)
    trainer.train()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='egemaps estimator')
    parser.add_argument("--train_config", type=str, required=True)
    conf_args = parser.parse_args()
    with open(conf_args.train_config, 'rt') as f:
        args = argparse.Namespace()
        args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
        args.__dict__.update(conf_args.__dict__)
        args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # print(args)
    set_seed(args.seed)
    main(args)
