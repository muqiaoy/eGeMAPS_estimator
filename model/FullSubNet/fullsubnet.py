import torch
from torch.nn import functional

from .feature import drop_band
from .base_model import BaseModel
from .sequence_model import SequenceModel


from ..Demucs.utils import capture_init


class FullSubNet(BaseModel):
    @capture_init
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 sample_rate=16000,
                 ):
        """
        FullSubNet model (cIRM mask)
        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.sample_rate = sample_rate
        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag, dropping_band=True):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram
        Returns:
            The real part and imag part of the enhanced spectrogram
        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        fb_output_unfolded = self.unfold(fb_output, num_neighbors=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbors=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation.
        # These will be updated to the paper later.
        if batch_size > 1 and dropping_band:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        return output


    def stft(self, y, n_fft=512, hop_length=256, win_length=512):
        """
        Wrapper of the official torch.stft for single-channel and multi-channel
        Args:
            y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
            n_fft: num of FFT
            hop_length: hop length
            win_length: hanning window size
        Shapes:
            mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]
        Returns:
            mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
        """
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)

        complex_stft = torch.stft(y, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=y.device),
                                  return_complex=True)
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

        mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
        real, imag = complex_stft.real, complex_stft.imag
        return mag, phase, real, imag

    def istft(self, features, n_fft=512, hop_length=256, win_length=512, length=None, input_type="complex"):
        """
        Wrapper of the official torch.istft
        Args:
            features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
            n_fft: num of FFT
            hop_length: hop length
            win_length: hanning window size
            length: expected length of istft
            use_mag_phase: use mag and phase as the input ("features")
        Returns:
            single-channel speech of shape [B, T]
        """
        if input_type == "real_imag":
            # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert isinstance(features, torch.ComplexType)
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError("Only 'real_imag', 'complex', and 'mag_phase' are supported")

        return torch.istft(features, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=features.device),
                           length=length)


if __name__ == "__main__":
    with torch.no_grad():
        noisy_mag = torch.rand(1, 1, 257, 63)
        model = Model(
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        print(model(noisy_mag).shape)