import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbedding_FreqComplex(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DataEmbedding_FreqComplex, self).__init__()
        self.linear_real = nn.Linear(2 * seq_len, d_model)
        self.linear_imag = nn.Linear(2 * seq_len, d_model)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.fft(x, n=2 * L)
        x_real = self.linear_real(x_fft.real)
        x_imag = self.linear_imag(x_fft.imag)
        x = torch.complex(x_real, x_imag)
        return x


class DataEmbedding_FreqInterpolate(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DataEmbedding_FreqInterpolate, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.fft(x, n=2 * L)
        x_fft_resampled = self.resample_fft(x_fft, self.d_model)
        return x_fft_resampled

    def resample_fft(self, x_fft, new_length):
        real_part = x_fft.real
        imag_part = x_fft.imag
        real_interpolated = F.interpolate(real_part, size=new_length, mode='linear', align_corners=False)
        imag_interpolated = F.interpolate(imag_part, size=new_length, mode='linear', align_corners=False)
        x_fft_resampled = torch.complex(real_interpolated, imag_interpolated)
        return x_fft_resampled


class DataEmbedding_Freq_FourierInterpolate(nn.Module):
    def __init__(self, seq_len, d_model, c_in):
        super(DataEmbedding_Freq_FourierInterpolate, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.scalars = nn.Parameter(torch.ones(c_in, d_model), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(c_in, d_model, dtype=torch.cfloat), requires_grad=True)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.rfft(x, n=2 * L)
        x_fft_resampled = self.fourier_interpolate(x_fft, self.d_model)
        x_fft_resampled = x_fft_resampled * self.scalars + self.bias
        return x_fft_resampled

    def fourier_interpolate(self, x_fft, new_length):
        B, N, L = x_fft.shape
        if new_length > L:
            resampled_data = torch.zeros(B, N, new_length, dtype=torch.cfloat,
                                   device=x_fft.device) + 0.0001
            resampled_data[:, :, :L] = x_fft
        else:
            resampled_data = x_fft[:, :, :new_length]
        return resampled_data
