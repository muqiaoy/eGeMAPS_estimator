# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os
import re
import librosa
from tqdm import tqdm

import numpy as np
import torch
import torchaudio

from .audioset import Audioset, find_audio_files

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="dns"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    def __init__(self, dataPath, num_files=None, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, egemaps_path=None, egemaps_lld_path=None, spec_path=None):
        """__init__.
        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(dataPath, 'noisy.json')
        clean_json = os.path.join(dataPath, 'clean.json')
        print("Loading data from data path %s" % dataPath)
        if not os.path.exists(noisy_json) or not os.path.exists(clean_json):
            print("Generating json files for data path %s" % dataPath)
            noisy = find_audio_files(os.path.join(dataPath, "noisy"))
            clean = find_audio_files(os.path.join(dataPath, "clean"))
            with open(noisy_json, 'w') as f:
                json.dump(noisy, f)
            with open(clean_json, 'w') as f:
                json.dump(noisy, f)

        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'egemaps_path': egemaps_path, 'egemaps_lld_path': egemaps_lld_path, 'spec_path': spec_path}
        if num_files is not None and num_files < len(noisy):
            noisy = noisy[:num_files]
            clean = clean[:num_files]
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        # If egemaps_path is not None, __getitem__() will output one more object which is the egemaps features
        self.egemaps_path = egemaps_path
        self.spec_path = spec_path
        self.egemaps_lld_path = egemaps_lld_path

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        
        return self.noisy_set[index], self.clean_set[index], self.egemaps[index] if self.egemaps_path is not None else torch.Tensor([-1]), self.spec[index] if self.spec_path is not None else torch.Tensor([-1]), self.egemaps_lld[index] if self.egemaps_lld_path is not None else torch.Tensor([-1])

    def __len__(self):
        return len(self.noisy_set)