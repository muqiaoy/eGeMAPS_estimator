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
import opensmile
import librosa

import numpy as np
import torch

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
    def __init__(self, noisy_dir, clean_dir, num_files=None, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, egemaps_path=None, egemaps_lld_path=None, spec_path=None):
        """__init__.
        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy = find_audio_files(noisy_dir)
        clean = find_audio_files(clean_dir)
        print(clean_dir)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        if num_files is not None and num_files < len(noisy):
            noisy = noisy[:num_files]
            clean = clean[:num_files]
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        # If egemaps_path is not None, __getitem__() will output one more object which is the egemaps features
        self.egemaps_path = egemaps_path
        self.spec_path = spec_path
        self.egemaps_lld_path = egemaps_lld_path


        # generate the egemaps features for the 1st time if it does not exist
        if egemaps_path is not None:
            if not os.path.exists(egemaps_path):
                self.smile_F = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals)
                self.egemaps = torch.zeros(len(self.clean_set), len(self.smile_F.feature_names))
                for i in range(len(self.clean_set)):
                    self.egemaps[i] = torch.from_numpy(self.smile_F.process_signal(self.clean_set[i], sampling_rate=sample_rate).values)
                if not os.path.exists("egemaps_functionals"):
                    os.makedirs("egemaps_functionals")
                np.save(os.path.join("egemaps_functionals", os.path.basename(egemaps_path)), self.egemaps.numpy())
            else:
                self.egemaps = torch.from_numpy(np.load(egemaps_path))
                if num_files is not None and num_files < len(self.egemaps):
                    self.egemaps = self.egemaps[:num_files]
                assert len(self.egemaps) == len(self.clean_set), \
                    "There is a mismatch between the length of the saved egemaps features (%s) and the dataset (%s). You may want to regenerate the features." % (len(self.egemaps), len(self.clean_set))

            self.egemaps = torch.nn.functional.normalize(self.egemaps)

        # print(self.clean_set[0].shape[-1] // 160 - 4)
        if egemaps_lld_path is not None:
            if not os.path.exists(egemaps_lld_path):
                self.smile_F = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
                print(self.clean_set[0].shape[-1] // 160 - 4)
                self.egemaps_lld = torch.zeros(len(self.clean_set), self.clean_set[0].shape[-1] // 160 - 4, len(self.smile_F.feature_names))
                for i in range(len(self.clean_set)):
                    self.egemaps_lld[i] = torch.from_numpy(self.smile_F.process_signal(self.clean_set[i], sampling_rate=sample_rate).values)
                if not os.path.exists("egemaps_lld"):
                    os.makedirs("egemaps_lld")
                np.save(os.path.join("egemaps_lld", os.path.basename(egemaps_lld_path)), self.egemaps_lld.numpy())
            else:
                self.egemaps_lld = torch.from_numpy(np.load(egemaps_lld_path))
                if num_files is not None and num_files < len(self.egemaps_lld):
                    self.egemaps_lld = self.egemaps_lld[:num_files]
                assert len(self.egemaps_lld) == len(self.clean_set), \
                    "There is a mismatch between the length of the saved egemaps features (%s) and the dataset (%s). You may want to regenerate the features." % (len(self.egemaps_lld), len(self.clean_set))

            self.egemaps_lld = torch.nn.functional.normalize(self.egemaps_lld)

        if spec_path is not None:
            if not os.path.exists(spec_path):
                self.spec = torch.zeros(len(self.clean_set), 128, 938 if self.clean_set[0].shape[-1] == 480000 else 313)
                for i in range(len(self.clean_set)):
                    self.spec[i] = torch.from_numpy(librosa.feature.melspectrogram(y=self.clean_set[i].numpy(), sr=sample_rate))
                if not os.path.exists("spec"):
                    os.makedirs("spec")
                np.save(os.path.join("spec", os.path.basename(spec_path)), self.spec.numpy())
            else:
                self.spec = torch.from_numpy(np.load(spec_path))
                if num_files is not None and num_files < len(self.spec):
                    self.spec = self.spec[:num_files]
                assert len(self.spec) == len(self.clean_set), \
                    "There is a mismatch between the length of the saved spectrograms (%s) and the dataset (%s). You may want to regenerate the features." % (len(self.spec), len(self.clean_set))


            self.spec = torch.nn.functional.normalize(self.spec)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        
        return self.noisy_set[index], self.clean_set[index], self.egemaps[index] if self.egemaps_path is not None else torch.Tensor(-1), self.spec[index] if self.spec_path is not None else torch.Tensor(-1), self.egemaps_lld[index] if self.egemaps_lld_path is not None else torch.Tensor(-1)

    def __len__(self):
        return len(self.noisy_set)