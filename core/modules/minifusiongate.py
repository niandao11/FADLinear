import math
import torch
import torch.nn as nn





class HorizonSoftmaxGate(nn.Module):
    def __init__(self, pred_len: int, init_cov: float = 0.20):  
        super().__init__()
        steps = torch.linspace(0.0, 1.0, pred_len)  
        
        w_season = 0.70 * (1.0 - steps) + 0.05
        w_trend = 0.70 * steps + 0.05
        w_cov = torch.full_like(steps, init_cov)  
        w = torch.stack([w_season, w_trend, w_cov], dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        self.logits = nn.Parameter(torch.log(w.clamp_min(1e-8)))  

    def forward(self, pred_season: torch.Tensor, pred_trend: torch.Tensor, pred_cov: torch.Tensor) -> torch.Tensor:
        
        weights = torch.softmax(self.logits, dim=-1).unsqueeze(0).unsqueeze(-1)  
        comps = torch.stack([pred_season, pred_trend, pred_cov], dim=2)          
        return (weights * comps).sum(dim=2)                                      





class HorizonSoftmaxGate2(nn.Module):
    def __init__(self, pred_len: int):
        super().__init__()
        steps = torch.linspace(0.0, 1.0, pred_len)  
        
        w_season = 0.80 * (1.0 - steps) + 0.10
        w_trend  = 0.80 * steps + 0.10
        w = torch.stack([w_season, w_trend], dim=-1)  
        w = w / w.sum(dim=-1, keepdim=True)
        self.logits = nn.Parameter(torch.log(w.clamp_min(1e-8)))  

    def forward(self, pred_season: torch.Tensor, pred_trend: torch.Tensor) -> torch.Tensor:
        
        weights = torch.softmax(self.logits, dim=-1).unsqueeze(0).unsqueeze(-1)  
        comps = torch.stack([pred_season, pred_trend], dim=2)                    
        return (weights * comps).sum(dim=2)


class TrendDominanceHorizonGate2(nn.Module):
    def __init__(self, pred_len: int, init_gamma: float = 0.01, max_gamma: float = 2.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.max_gamma = float(max_gamma)

        steps = torch.linspace(0.0, 1.0, pred_len)
        w_season = 0.80 * (1.0 - steps) + 0.10
        w_trend = 0.80 * steps + 0.10
        w = torch.stack([w_season, w_trend], dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        self.logits = nn.Parameter(torch.log(w.clamp_min(1e-8)))

        init_gamma = min(max(float(init_gamma), 1e-6), self.max_gamma - 1e-6)
        init_prob = init_gamma / self.max_gamma
        init_raw = math.log(init_prob / (1.0 - init_prob))
        self.gamma_raw = nn.Parameter(torch.full((pred_len,), init_raw))

    def _gamma(self) -> torch.Tensor:
        return self.max_gamma * torch.sigmoid(self.gamma_raw)

    def weights(self, trend_ratio: torch.Tensor) -> torch.Tensor:
        if trend_ratio.dim() == 1:
            trend_ratio = trend_ratio[:, None]
        elif trend_ratio.dim() == 3:
            trend_ratio = trend_ratio.squeeze(-1)

        trend_ratio = trend_ratio.clamp(0.0, 1.0)
        state = 2.0 * trend_ratio - 1.0                         
        delta = state * self._gamma().view(1, self.pred_len)     

        logits = self.logits.unsqueeze(0).expand(delta.size(0), -1, -1).clone()
        logits[:, :, 0] = logits[:, :, 0] - delta
        logits[:, :, 1] = logits[:, :, 1] + delta
        return torch.softmax(logits, dim=-1)                     

    def forward(
        self,
        pred_season: torch.Tensor,
        pred_trend: torch.Tensor,
        trend_ratio: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.weights(trend_ratio).unsqueeze(-1)        
        comps = torch.stack([pred_season, pred_trend], dim=2)    
        return (weights * comps).sum(dim=2)


class EnergyContrastHorizonGate2(nn.Module):
    def __init__(self, pred_len: int, init_beta: float = 0.0, max_beta: float = 2.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.max_beta = float(max_beta)

        steps = torch.linspace(0.0, 1.0, pred_len)
        w_season = 0.80 * (1.0 - steps) + 0.10
        w_trend = 0.80 * steps + 0.10
        w = torch.stack([w_season, w_trend], dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        self.logits = nn.Parameter(torch.log(w.clamp_min(1e-8)))

        init_beta = max(min(float(init_beta), self.max_beta - 1e-6), -self.max_beta + 1e-6)
        init_raw = math.atanh(init_beta / self.max_beta)
        self.beta_raw = nn.Parameter(torch.full((pred_len,), init_raw))

    def _beta(self) -> torch.Tensor:
        return self.max_beta * torch.tanh(self.beta_raw)

    def weights(self, energy_contrast: torch.Tensor) -> torch.Tensor:
        if energy_contrast.dim() == 1:
            energy_contrast = energy_contrast[:, None]
        elif energy_contrast.dim() == 3:
            energy_contrast = energy_contrast.squeeze(-1)

        state = energy_contrast.clamp(-1.0, 1.0)                 
        delta = state * self._beta().view(1, self.pred_len)      

        logits = self.logits.unsqueeze(0).expand(delta.size(0), -1, -1).clone()
        logits[:, :, 0] = logits[:, :, 0] - delta
        logits[:, :, 1] = logits[:, :, 1] + delta
        return torch.softmax(logits, dim=-1)                     

    def forward(
        self,
        pred_season: torch.Tensor,
        pred_trend: torch.Tensor,
        energy_contrast: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.weights(energy_contrast).unsqueeze(-1)    
        comps = torch.stack([pred_season, pred_trend], dim=2)    
        return (weights * comps).sum(dim=2)
