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

import torch
import torch.nn as nn

from model import *
from model.vae import VAE
from model.m5 import M5
from dataset import distrib
from dataset.dataset import NoisyCleanSet
from trainer.trainer_est import Trainer_est


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

    if args.model == 'VAE':
        estimator = VAE(**args.vae)

    elif args.model == 'M5':
        estimator = M5(**args.m5)

    else:
        raise NotImplementedError(args.model)
        
    length = int(args.segment * args.fs)
    stride = int(args.stride * args.fs)
    tr_dir = os.path.join(args.dataPath, args.trainPath)
    tr_dataset = NoisyCleanSet(tr_dir, num_files=args.num_train_files, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_train_path, egemaps_lld_path=args.egemaps_lld_train_path, spec_path=args.spec_train_path)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Total number of train segments: %s" % len(tr_dataset))
    cv_dir = os.path.join(args.dataPath, args.validPath)
    cv_dataset = NoisyCleanSet(cv_dir, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_valid_path, egemaps_lld_path=args.egemaps_lld_valid_path, spec_path=args.spec_valid_path)
    cv_loader = distrib.loader(
        cv_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    print("Total number of valid segments: %s" % len(cv_dataset))
    tt_dir = os.path.join(args.dataPath, args.testPath)
    tt_dataset = NoisyCleanSet(tt_dir, matching=args.matching, sample_rate=args.fs, egemaps_path=args.egemaps_test_path, egemaps_lld_path=args.egemaps_lld_test_path, spec_path=args.spec_test_path)
    tt_loader = distrib.loader(
        tt_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    print("Total number of test segments: %s" % len(tt_dataset))

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        estimator.cuda()
        # estimator = nn.DataParallel(estimator, device_ids=list(range(args.ngpu)))

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(estimator.parameters(), lr=float(args.lr), betas=(0.9, args.beta2))
    else:
        raise NotImplementedError
        
    trainer = Trainer_est(data, estimator, optimizer, args)
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
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.ngpu == -1:
            args.ngpu = torch.cuda.device_count()

    print(args)
    set_seed(args.seed)
    main(args)