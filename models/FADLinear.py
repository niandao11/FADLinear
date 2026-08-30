import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules.minifusiongate import EnergyContrastHorizonGate2
from core.modules.RevIN import RevIN


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(getattr(configs, "seq_len", 336))
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(getattr(configs, "enc_in", 7))
        self.target_idx = int(getattr(configs, "target_idx", self.enc_in - 1))

        self.revin_ot = RevIN(
            num_features=1,
            eps=1e-5,
            affine=True,
            subtract_last=bool(getattr(configs, "subtract_last", False)),
        )

        
        
        self.sigma_min = float(getattr(configs, "fad_sigma_min", 0.50))
        self.sigma_max = float(getattr(configs, "fad_sigma_max", 2.50))
        init_sigma = float(getattr(configs, "fad_init_sigma", 1.00))
        init_sigma = min(max(init_sigma, self.sigma_min + 1e-6), self.sigma_max - 1e-6)
        init_ratio = (init_sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.sigma_raw = nn.Parameter(init_logit.clone().detach())
        self.register_buffer("_kernel_pos", torch.arange(-2, 3, dtype=torch.float32))
        self.register_buffer("_eps", torch.tensor(1e-6))
        self.energy_contrast_clip = float(getattr(configs, "fad_energy_contrast_clip", 6.0))

        
        self.trend_len = (self.seq_len + 1) // 2

        
        self.head_season = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.head_trend  = nn.Linear(self.trend_len, self.pred_len, bias=False)

        
        self.gate = EnergyContrastHorizonGate2(
            self.pred_len,
            init_beta=float(getattr(configs, "fad_gate_init_beta", 0.0)),
            max_beta=float(getattr(configs, "fad_gate_max_beta", 2.0)),
        )

    def _kernel(self) -> torch.Tensor:
        sigma_unit = torch.sigmoid(self.sigma_raw)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * sigma_unit
        pos = self._kernel_pos.to(dtype=sigma.dtype, device=sigma.device)
        k = torch.exp(-(pos ** 2) / (2.0 * sigma ** 2 + self._eps))
        k = k / (k.sum() + self._eps)
        return k.view(1, 1, 5)

    def _kernel_sigma(self) -> torch.Tensor:
        sigma_unit = torch.sigmoid(self.sigma_raw)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * sigma_unit

    def _down2(self, x: torch.Tensor) -> torch.Tensor:
        k = self._kernel()
        x = F.pad(x, (2, 2), mode="reflect")
        return F.conv1d(x, k, stride=2)

    def _up2(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        k = self._kernel() * 2.0
        y = F.conv_transpose1d(x, k, stride=2, padding=2, output_padding=1)

        if y.shape[-1] > target_size:
            y = y[..., :target_size]
        elif y.shape[-1] < target_size:
            y = F.pad(y, (0, target_size - y.shape[-1]))
        return y

    def forward(self, batch_x, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        B, L, D = batch_x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"  
        assert D >= self.enc_in, f"Expected D>={self.enc_in}, got {D}"

        x = batch_x[:, :, :self.enc_in]  

        
        x_ot = x[:, :, self.target_idx:self.target_idx + 1]   
        x_ot = self.revin_ot(x_ot, mode="norm")               
        x_ot = x_ot.transpose(1, 2).contiguous()              

        
        trend_down = self._down2(x_ot)                        
        trend_up   = self._up2(trend_down, L)                 
        season     = x_ot - trend_up                          

        
        trend_var = trend_up.var(dim=-1, unbiased=False)
        season_var = season.var(dim=-1, unbiased=False)
        energy_contrast = torch.log((trend_var + self._eps) / (season_var + self._eps))
        energy_contrast = energy_contrast.clamp(
            -self.energy_contrast_clip, self.energy_contrast_clip
        ) / self.energy_contrast_clip

        
        pred_season = self.head_season(season).transpose(1, 2)     
        pred_trend  = self.head_trend(trend_down).transpose(1, 2)  

        
        y = self.gate(
            pred_season=pred_season,
            pred_trend=pred_trend,
            energy_contrast=energy_contrast,
        )  

        
        y = self.revin_ot(y, mode="denorm")
        return y
