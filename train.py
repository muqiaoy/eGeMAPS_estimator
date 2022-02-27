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

from model import *
from data import distrib
from data.dataset import NoisyCleanSet
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
    if args.model == 'NSNet2':
        model = NSNet2(modelfile=args.modelPath, cfg=args.cfg)
        logging.info("Loaded checkpoint from %s" % args.modelPath)
    elif args.model == 'Demucs':
        model = Demucs(**args.demucs, sample_rate=args.fs)
        state_dict = torch.load(args.modelPath)
        model.load_state_dict(state_dict)
        logging.info("Loaded checkpoint from %s" % args.modelPath)
        
        length = int(args.segment * args.fs)
        stride = int(args.stride * args.fs)
        tr_noisy_dir = os.path.join(args.dataPath, "synthesized/noisy")
        tr_clean_dir = os.path.join(args.dataPath, "synthesized/clean")
        tr_dataset = NoisyCleanSet(tr_noisy_dir, tr_clean_dir, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs)
        tr_loader = distrib.loader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print("Total number of train frames: %s" % len(tr_dataset))
        tt_noisy_dir = os.path.join(args.dataPath, "test_set/synthetic/no_reverb/noisy")
        tt_clean_dir = os.path.join(args.dataPath, "test_set/synthetic/no_reverb/clean")
        tt_dataset = NoisyCleanSet(tt_noisy_dir, tt_clean_dir, length=length, stride=stride, pad=args.pad, matching=args.matching, sample_rate=args.fs)
        tt_loader = distrib.loader(
            tt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        print("Total number of test frames: %s" % len(tt_dataset))

        data = {"tr_loader": tr_loader, "cv_loader": None, "tt_loader": tt_loader}

        if torch.cuda.is_available():
            model.cuda()

        if args.optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, args.beta2))
        else:
            raise NotImplementedError
            
        trainer = Trainer(data, model, optimizer, args)
        trainer.train()

    else:
        raise NotImplementedError


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

    print(args)
    set_seed(args.seed)
    main(args)