# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys

import torch
import torchaudio

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from dataset.audioset import Audioset, find_audio_files
from dataset import distrib
from . import load_pretrained
from model import *
from model.FullSubNet.mask import build_complex_ideal_ratio_mask, decompress_cIRM

from .utils import LogProgress

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    load_pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.streaming:
        raise NotImplementedError
    else:
        with torch.no_grad():
            if isinstance(model, Demucs):
                estimate, _ = model(noisy)
                estimate = (1 - args.dry) * estimate + args.dry * noisy
            elif isinstance(model, FullSubNet):
                # full band crm mask

                noisy_mag, noisy_phase, noisy_real, noisy_imag = model.stft(torch.squeeze(noisy, dim=1))

                noisy_mag = noisy_mag.unsqueeze(1)
                pred_crm = model(noisy_mag, dropping_band=False)
                pred_crm = pred_crm.permute(0, 2, 3, 1)

                pred_crm = decompress_cIRM(pred_crm)
                enhanced_real = pred_crm[..., 0] * noisy_real - pred_crm[..., 1] * noisy_imag
                enhanced_imag = pred_crm[..., 1] * noisy_real + pred_crm[..., 0] * noisy_imag
                estimate = model.istft((enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag")
                estimate = torch.unsqueeze(estimate, dim=1)
                # estimate = enhanced.detach().squeeze(0).cpu().numpy()
            else:
                raise NotImplementedError
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        # write(noisy, filename + "_noisy.wav", sr=sr)
        write(estimate, filename + ".wav", sr=sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(args, sample_rate, channels=1):
    if hasattr(args, 'dset'):
        paths = args.dset
    else:
        paths = args
    if hasattr(paths, "noisy_json") and paths.noisy_json:
        with open(paths.noisy_json) as f:
            files = json.load(f)
    elif hasattr(paths, "noisy_dir") and paths.noisy_dir:
        files = find_audio_files(paths.noisy_dir)
    elif hasattr(paths, "testPath") and paths.testPath:
        files = find_audio_files(os.path.join(paths.dataPath, paths.testPath, "noisy"))
    else:
        logger.warning(
            "Small sample set was not provided by either noisy_dir or noisy_json. "
            "Skipping enhancement.")
        return None
    return Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, args):
    estimate = get_estimate(model, noisy_signals, args)
    save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)


def enhance(args, model=None, local_out_dir=None):
    # Load model
    if not model:
        model = load_pretrained.get_model(args).to(args.device)
    model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir

    dset = get_dataset(args, model.sample_rate)
    if dset is None:
        return
    loader = distrib.loader(dset, batch_size=1)

    if distrib.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    distrib.barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, noisy_signals, filenames, out_dir, args))
            else:
                # Forward
                estimate = get_estimate(model, noisy_signals, args)
                save_wavs(estimate, noisy_signals, filenames, out_dir, sr=model.sample_rate)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()


if __name__ == "__main__":

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    enhance(args, local_out_dir=args.out_dir)