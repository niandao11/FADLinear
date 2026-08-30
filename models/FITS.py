import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.h_order = configs.fits_h_order
        self.base_t = configs.fits_base_t
        self.output_len = self.seq_len + self.pred_len
        self.length_ratio = self.output_len / self.seq_len
        self.cut_freq = (self.seq_len // self.base_t + 1) * self.h_order + 10
        self.output_cut_freq = int(self.cut_freq * self.length_ratio)
        input_bins = self.seq_len // 2 + 1
        output_bins = self.output_len // 2 + 1
        if self.cut_freq > input_bins or self.output_cut_freq > output_bins:
            raise ValueError(
                f"Invalid FITS frequency sizes: input={self.cut_freq}/{input_bins}, "
                f"output={self.output_cut_freq}/{output_bins}."
            )
        if self.individual:
            self.frequency_upsampler = nn.ModuleList([
                nn.Linear(self.cut_freq, self.output_cut_freq).to(torch.cfloat)
                for _ in range(self.channels)
            ])
        else:
            self.frequency_upsampler = nn.Linear(
                self.cut_freq, self.output_cut_freq
            ).to(torch.cfloat)

    def _forward_full(self, x_enc):
        if x_enc.dtype != torch.float32:
            raise TypeError(f"FITS supports float32 input, got {x_enc.dtype}.")
        if x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(
                f"FITS expected [B,{self.seq_len},{self.channels}], got {tuple(x_enc.shape)}."
            )
        mean = x_enc.mean(dim=1, keepdim=True)
        variance = x_enc.var(dim=1, keepdim=True, unbiased=True)
        scale = torch.sqrt(variance + 1e-5)
        normalized = (x_enc - mean) / scale
        spectrum = torch.fft.rfft(normalized, dim=1)[:, :self.cut_freq, :]
        if self.individual:
            upsampled = torch.zeros(
                spectrum.size(0), self.output_cut_freq, self.channels,
                dtype=torch.complex64, device=spectrum.device
            )
            for channel, layer in enumerate(self.frequency_upsampler):
                upsampled[:, :, channel] = layer(spectrum[:, :, channel])
        else:
            upsampled = self.frequency_upsampler(
                spectrum.permute(0, 2, 1)
            ).permute(0, 2, 1)
        output_spectrum = torch.zeros(
            spectrum.size(0), self.output_len // 2 + 1, self.channels,
            dtype=torch.complex64, device=spectrum.device
        )
        output_spectrum[:, :self.output_cut_freq, :] = upsampled
        output = torch.fft.irfft(output_spectrum, n=self.output_len, dim=1)
        output = output * self.length_ratio
        return output * scale + mean

    def forward(self, x_enc, return_full_sequence=False):
        full_sequence = self._forward_full(x_enc)
        if return_full_sequence:
            return full_sequence
        return full_sequence[:, -self.pred_len:, :]