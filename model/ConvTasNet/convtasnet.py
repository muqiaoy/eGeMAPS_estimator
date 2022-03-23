"""Enhancement model module."""
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.separator.tcn_separator import TCNSeparator
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")

EPS = torch.finfo(torch.get_default_dtype()).eps


class ConvTasNet(nn.Module):
    """Speech enhancement or separation Frontend model"""

    def __init__(
        self,
        encoder_conf: Dict,
        separator_conf: Dict,
        decoder_conf: Dict,
        criterions: Dict,
        sample_rate: int,
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = ConvEncoder(**encoder_conf)
        self.separator = TCNSeparator(**separator_conf)
        self.decoder = ConvDecoder(**decoder_conf)
        
        criterion = SISNRLoss(**criterions["conf"])
        loss_wrapper = PITSolver(
            criterion=criterion, **criterions["wrapper_conf"]
        )

        self.sample_rate = sample_rate


        self.loss_wrapper = loss_wrapper
        self.num_spk = self.separator.num_spk
        self.num_noise_type = getattr(self.separator, "num_noise_type", 1)

        # get mask type for TF-domain models
        # (only used when loss_type="mask_*") (deprecated, keep for compatibility)
        self.mask_type = mask_type.upper() if mask_type else None

        # get loss type for model training (deprecated, keep for compatibility)
        self.loss_type = loss_type

        # whether to compute the TF-domain loss
        # while enforcing STFT consistency (deprecated, keep for compatibility)
        self.stft_consistency = stft_consistency

        # for multi-channel signal (deprecated, keep for compatibility)
        self.ref_channel = getattr(self.separator, "ref_channel", -1)

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(self.num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                dereverb_speech_ref
            )
            # (Batch, N, samples) or (Batch, N, samples, channels)
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        batch_size = speech_mix.shape[0]
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()]
        speech_ref = speech_ref.unbind(dim=1)

        speech_mix = speech_mix[:, : speech_lengths.max()]

        # model forward
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_pre, flens, others = self.separator(feature_mix, flens)
        if feature_pre is not None:
            speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
        else:
            # some models (e.g. neural beamformer trained with mask loss)
            # do not predict time-domain signal in the training stage
            speech_pre = None

        loss = 0.0
        o = {}
        criterion = loss_wrapper.criterion
        if isinstance(criterion, TimeDomainLoss):
            if speech_ref[0].dim() == 3:
                # For multi-channel reference,
                # only select one channel as the reference
                speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
            # for the time domain criterions
            l, _, _ = loss_wrapper(speech_ref, speech_pre, o)
        else:
            raise NotImplementedError
        loss += l * loss_wrapper.weight

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        return speech_ref, loss

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}


    def enhance(self, speech_mix, fs=16000):
        """Inference
        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [separated_audio1, separated_audio2, ...]
        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        # lengths: (B,)
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]  # B x T [x C]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Enhancement/Separation Forward
                feats, f_lens = self.enh_model.encoder(speech_seg, lengths_seg)
                feats, _, _ = self.enh_model.separator(feats, f_lens)
                processed_wav = [
                    self.enh_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                if speech_seg.dim() > 2:
                    # multi-channel speech
                    speech_seg_ = speech_seg[:, self.ref_channel]
                else:
                    speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    # normalize the scale to match the input mixture scale
                    mix_energy = torch.sqrt(
                        torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
                    )
                    enh_energy = torch.sqrt(
                        torch.mean(
                            sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True
                        )
                    )
                    processed_wav = [
                        w * (mix_energy / enh_energy) for w in processed_wav
                    ]
                # List[torch.Tensor(num_spk, B, T)]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            # c. Stitch the enhanced segments together
            waves = enh_waves[0]
            for i in range(1, num_segments):
                # permutation between separated streams in last and current segments
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                # repermute separated streams in current segment
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                # overlap-and-add (average over the overlapped part)
                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            # ensure the stitched length is same as input
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            # b. Enhancement/Separation Forward
            feats, f_lens = self.enh_model.encoder(speech_mix, lengths)
            feats, _, _ = self.enh_model.separator(feats, f_lens)
            waves = [self.enh_model.decoder(f, lengths)[0] for f in feats]

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9)
                for w in waves
            ]  # list[(batch, sample)]
        waves = torch.stack(waves)
        print(waves.shape)
        assert False

        return waves
