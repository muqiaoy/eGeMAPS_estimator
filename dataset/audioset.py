# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys
import multiprocessing
from tqdm import tqdm
from glob import glob
import numpy as np

import torchaudio
import torch
from torch.nn import functional as F

from .dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    print("Loading data")
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    print("\n")
    meta.sort()
    return meta


def get_egemap(file, file_length, examples, output_path, smile, length, stride, sample_rate, level):

    num_frames = 0
    offset = 0
    if level == 'func':
        egemaps = np.zeros((examples, len(smile.feature_names)))
    elif level == 'lld':
        if length is not None:
            egemaps = np.zeros((examples, length // 160 - 4, len(smile.feature_names)))
        else:
            egemaps = np.zeros((examples, file_length // 160 - 4, len(smile.feature_names)))
    else:
        raise NotImplementedError
    for seg_idx in range(examples):
        if length is not None:
            offset = stride * seg_idx
            num_frames = length
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            seg, sr = torchaudio.load(str(file),
                                    frame_offset=offset,
                                    num_frames=num_frames or -1)
        else:
            seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
        egemaps[seg_idx] = smile.process_signal(seg, sampling_rate=sample_rate).values
    np.save(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))), egemaps)


def get_spec(file, file_length, examples, output_path, spectrogram, length, stride, sample_rate):

    num_frames = 0
    offset = 0
    if length is not None:
        spec = np.zeros((examples, spectrogram.win_length // 2 + 1, length // spectrogram.hop_length + 1))
    else:
        spec = np.zeros((examples, spectrogram.win_length // 2 + 1, file_length // spectrogram.hop_length + 1))
    for seg_idx in range(examples):
        if length is not None:
            offset = stride * seg_idx
            num_frames = length
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            seg, sr = torchaudio.load(str(file),
                                    frame_offset=offset,
                                    num_frames=num_frames or -1)
        else:
            seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
        spec[seg_idx] = spectrogram(seg)
    np.save(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))), spec)


def execute_multiprocess(files, num_examples, output_path, smile, length, stride, sample_rate, level):
    
    PROCESSES = 32
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        in_args = [(file, file_length, examples, output_path, smile, length, stride, sample_rate, level) 
                for (file, file_length), examples in zip(files, num_examples) if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))))]
        
        jobs = [pool.apply_async(get_egemap, in_arg) for in_arg in in_args]
        
        for j in tqdm(jobs):
            j.get()
            
    return None


def execute_multiprocess_spec(files, num_examples, output_path, spectrogram, length, stride, sample_rate):
    
    PROCESSES = 32
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        in_args = [(file, file_length, examples, output_path, spectrogram, length, stride, sample_rate) 
                for (file, file_length), examples in zip(files, num_examples) if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))))]
        
        jobs = [pool.apply_async(get_spec, in_arg) for in_arg in in_args]
        
        for j in tqdm(jobs):
            j.get()
            
    return None


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False, egemaps_path=None, egemaps_lld_path=None, spec_path=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        self.with_path = with_path
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

        
        self.egemaps_path = egemaps_path
        self.egemaps_lld_path = egemaps_lld_path
        self.spec_path = spec_path
        # generate the egemaps features for the 1st time if it does not exist
        if egemaps_lld_path is not None:
            if not os.path.exists(egemaps_lld_path):
                os.makedirs(egemaps_lld_path)
            if len(glob(os.path.join(egemaps_lld_path, "*.npy"))) < len(self.files):
                print("eGeMAPS LLDs do not exist (%d/%d). Generating... This might take a while" % (len(glob(os.path.join(egemaps_lld_path, "*.npy"))), len(files)))
                import opensmile
                smile_lld = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
                execute_multiprocess(self.files, self.num_examples, egemaps_lld_path, smile_lld, self.length, self.stride, self.sample_rate, level='lld')
                # for (file, file_length), examples in tqdm(zip(self.files, self.num_examples), total=len(self.files)):
                #     num_frames = 0
                #     offset = 0
                #     if self.length is not None:
                #         egemaps_lld = np.zeros((examples, self.length // 160 - 4, len(smile_lld.feature_names)))
                #     else:
                #         egemaps_lld = np.zeros((examples, file_length // 160 - 4, len(smile_lld.feature_names)))
                #     for seg_idx in range(examples):
                #         if self.length is not None:
                #             offset = self.stride * seg_idx
                #             num_frames = self.length
                #         if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                #             seg, sr = torchaudio.load(str(file),
                #                                     frame_offset=offset,
                #                                     num_frames=num_frames or -1)
                #         else:
                #             seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
                #         egemaps_lld[seg_idx] = smile_lld.process_signal(seg, sampling_rate=sample_rate).values
                #     np.save(os.path.join(egemaps_lld_path, os.path.basename(file.replace(".wav", ".npy"))), egemaps_lld)

        if egemaps_path is not None:
            if not os.path.exists(egemaps_path):
                os.makedirs(egemaps_path)
            if len(glob(os.path.join(egemaps_path, "*.npy"))) < len(self.files):
                print("eGeMAPS functionals do not exist (%d/%d). Generating... This might take a while" % (len(glob(os.path.join(egemaps_path, "*.npy"))), len(files)))
                import opensmile
                smile_func = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals)
                execute_multiprocess(self.files, self.num_examples, egemaps_path, smile_func, self.length, self.stride, self.sample_rate, level='func')
                # for (file, _), examples in tqdm(zip(self.files, self.num_examples), total=len(self.files)):
                #     num_frames = 0
                #     offset = 0
                #     egemaps_func = np.zeros((examples, len(smile_func.feature_names)))
                #     for seg_idx in range(examples):
                #         if self.length is not None:
                #             offset = self.stride * seg_idx
                #             num_frames = self.length
                #         if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                #             seg, sr = torchaudio.load(str(file),
                #                                     frame_offset=offset,
                #                                     num_frames=num_frames or -1)
                #         else:
                #             seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
                #         egemaps_func[seg_idx] = smile_func.process_signal(seg, sampling_rate=sample_rate).values
                #     np.save(os.path.join(egemaps_path, os.path.basename(file.replace(".wav", ".npy"))), egemaps_func)


        if spec_path is not None:
            raise NotImplementedError
            if not os.path.exists(spec_path):
                os.makedirs(spec_path)
            if len(glob(os.path.join(spec_path, "*.npy"))) < len(self.files):
                print("spectrograms do not exist (%d/%d). Generating... This might take a while" % (len(glob(os.path.join(spec_path, "*.npy"))), len(files)))
                spectrogram = torchaudio.transforms.Spectrogram(hop_length=160)
                execute_multiprocess_spec(self.files, self.num_examples, spec_path, spectrogram, self.length, self.stride, self.sample_rate)
            # if not os.path.exists(spec_path):
            #     print("specrograms do not exist. Generating... This might take a while")
            #     self.spec = torch.zeros(len(self.clean_set), 201, 938 if self.clean_set[0].shape[-1] == 480000 else 313)
            #     for i in tqdm(range(len(self.clean_set))):
            #         self.spec[i] = spectrogram(self.clean_set[i])
            #         # self.spec[i] = torch.from_numpy(librosa.feature.melspectrogram(y=self.clean_set[i].numpy(), sr=sample_rate))
            #     if not os.path.exists("spec"):
            #         os.makedirs("spec")
            #     np.save(os.path.join("spec", os.path.basename(spec_path)), self.spec.numpy())
            # else:
            #     self.spec = torch.from_numpy(np.load(spec_path))
            #     if num_files is not None and num_files < len(self.spec):
            #         self.spec = self.spec[:num_files]
            #     assert len(self.spec) == len(self.clean_set), \
            #         "There is a mismatch between the length of the saved spectrograms (%s) and the dataset (%s). You may want to regenerate the features." % (len(self.spec), len(self.clean_set))


    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_channels}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))


            if self.egemaps_path is not None:
                egemaps_func = torch.from_numpy(np.load(os.path.join(self.egemaps_path, os.path.basename(file.replace(".wav", ".npy"))))[index:index+1]).float()
                egemaps_func = F.normalize(egemaps_func)
            else:
                egemaps_func = torch.Tensor([-1])
            if self.egemaps_lld_path is not None:
                egemaps_lld = torch.from_numpy(np.load(os.path.join(self.egemaps_lld_path, os.path.basename(file.replace(".wav", ".npy"))))[index:index+1]).float()
                egemaps_lld = F.normalize(egemaps_lld)
            else:
                egemaps_lld = torch.Tensor([-1])
            if self.spec_path is not None:
                spec = torch.from_numpy(np.load(os.path.join(self.spec_path, os.path.basename(file.replace(".wav", ".npy"))))[index:index+1]).float()
                spec = F.normalize(spec)
            else:
                spec = torch.Tensor([-1])
            if self.with_path:
                return out, file
            else:
                return out, egemaps_func, egemaps_lld, spec


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)