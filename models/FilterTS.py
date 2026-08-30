import torch
import torch.nn as nn
from core.modules.freq_embed import (DataEmbedding_FreqInterpolate,
                                      DataEmbedding_FreqComplex)
from core.modules.filter_layer import FrequencyDomainFilterLayer
from core.modules.complex_func import ComplexLayerNorm


class ComplexProjection(nn.Module):
    def __init__(self, d_model, freq_len):
        super(ComplexProjection, self).__init__()
        self.linear_real = nn.Linear(d_model, d_model)
        self.linear_imag = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model * 2, freq_len)

    def forward(self, x):
        real_part = self.linear_real(x.real) - self.linear_imag(x.imag)
        imag_part = self.linear_imag(x.real) + self.linear_real(x.imag)
        x = torch.cat((real_part, imag_part), dim=-1)
        x = self.linear_out(x)
        return x


class FilterTSFourierEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, c_in):
        super(FilterTSFourierEmbedding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.scalars = nn.Parameter(torch.ones(c_in, d_model), requires_grad=True)
        self.bias_real = nn.Parameter(torch.zeros(c_in, d_model), requires_grad=True)
        self.bias_imag = nn.Parameter(torch.zeros(c_in, d_model), requires_grad=True)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        _, _, length = x.size()
        x_fft = torch.fft.rfft(x, n=2 * length)
        x_fft_resampled = self.fourier_interpolate(x_fft, self.d_model)
        bias = torch.complex(self.bias_real, self.bias_imag)
        return x_fft_resampled * self.scalars.unsqueeze(0) + bias.unsqueeze(0)

    def fourier_interpolate(self, x_fft, new_length):
        batch_size, n_vars, freq_len = x_fft.shape
        if new_length > freq_len:
            resampled_data = torch.zeros(
                batch_size, n_vars, new_length, dtype=torch.cfloat, device=x_fft.device
            ) + 0.0001
            resampled_data[:, :, :freq_len] = x_fft
        else:
            resampled_data = x_fft[:, :, :new_length]
        return resampled_data


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_out = args.c_out
        self.c_in = args.enc_in
        self.d_model = args.d_model
        self.use_norm = getattr(args, 'use_norm', False)
        self.filter_type = getattr(args, 'filter_type', 'all')
        self.quantile = getattr(args, 'quantile', 0.9)
        self.bandwidth = getattr(args, 'bandwidth', 1)
        self.embedding = getattr(args, 'embedding', 'fourier_interpolate')
        self.top_K_static_freqs = getattr(args, 'top_K_static_freqs', 10)

        self.model = nn.ModuleList([FrequencyDomainFilterLayer(
            self.seq_len, self.d_model, self.c_in,
            filter_type=self.filter_type,
            bandwidth=self.bandwidth,
            top_K_static_freqs=self.top_K_static_freqs,
            quantile=self.quantile)
            for _ in range(args.e_layers)])

        if self.embedding == "fourier_interpolate":
            self.enc_embedding = FilterTSFourierEmbedding(self.seq_len, self.d_model, self.c_in)
        elif self.embedding == "interpolate":
            self.enc_embedding = DataEmbedding_FreqInterpolate(self.seq_len, self.d_model)
        else:
            self.enc_embedding = DataEmbedding_FreqComplex(self.seq_len, self.d_model)

        self.layer = args.e_layers
        self.layer_norm = ComplexLayerNorm(self.d_model)
        self.projection = ComplexProjection(self.d_model, args.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_in = self.layer_norm(self.enc_embedding(x_enc, x_mark_enc))
        for i in range(self.layer):
            enc_out = self.model[i](enc_in)
            enc_out = self.layer_norm(enc_out)

        dec_in = enc_out + enc_in
        dec_out = self.projection(dec_in)
        dec_out = dec_out.transpose(2, 1)[:, :, -self.c_out:]

        if self.use_norm:
            target_stdev = stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1)
            target_means = means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out * target_stdev
            dec_out = dec_out + target_means

        return dec_out[:, -self.pred_len:, :]
