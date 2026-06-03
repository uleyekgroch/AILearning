"""主动推理 / 自由能原理 — FEP 动作选择 + 概率预测 + 变分信念

核心思想：不是最大化奖励，而是最小化期望自由能。
期望自由能 G = -信息增益 - 工具价值 + 风险惩罚
统一好奇心（探索）和目标导向（利用）。
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Tuple


class ProbabilisticPredictor(nn.Module):
    """概率预测器 — 输出 (mean, log_variance) 而非点估计

    在 PredictiveCodingEngine 基础上增加不确定性估计。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, obs_dim)
        self.logvar_head = nn.Linear(hidden_dim, obs_dim)
        nn.init.xavier_uniform_(self.shared[0].weight)
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, obs: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (mean, log_variance)"""
        x = self.shared(torch.cat([obs, action], dim=-1))
        return self.mean_head(x), self.logvar_head(x)

    def predict(self, obs: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs, action)


class VariationalBelief:
    """变分信念 — Bayesian 精度加权信念更新

    跟踪每个维度的 posterior precision 和 surprise。
    """

    def __init__(self, dim: int, device: torch.device = None):
        self.dim = dim
        self.device = device or torch.device('cpu')
        self.log_var = torch.full((dim,), -1.0, device=self.device)
        self.mean = torch.zeros(dim, device=self.device)
        self._precision_history: List[float] = []
        self._surprise_history: List[float] = []

    def update(self, observation: torch.Tensor,
               predicted_mean: torch.Tensor,
               predicted_logvar: torch.Tensor,
               lr: float = 0.1) -> None:
        """精度加权信念更新"""
        pe = observation - predicted_mean
        predicted_var = torch.exp(predicted_logvar)
        prior_precision = torch.exp(-self.log_var)
        likelihood_precision = 1.0 / (predicted_var + 1e-8)

        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / (posterior_precision + 1e-8)
        new_mean = posterior_var * (
            prior_precision * self.mean + likelihood_precision * predicted_mean
        )
        self.mean = (1 - lr) * self.mean + lr * new_mean
        self.log_var = (1 - lr) * self.log_var + lr * torch.log(posterior_var + 1e-8)

        surprise = 0.5 * torch.sum(
            pe ** 2 * posterior_precision + torch.log(posterior_var + 1e-8)
        )
        self._surprise_history.append(surprise.item())
        self._precision_history.append(posterior_precision.mean().item())

    def get_precision(self) -> torch.Tensor:
        return torch.exp(-self.log_var)

    def get_surprise(self) -> float:
        if not self._surprise_history:
            return 0.0
        return self._surprise_history[-1]

    def get_avg_surprise(self) -> float:
        if not self._surprise_history:
            return 0.0
        recent = self._surprise_history[-50:]
        return sum(recent) / len(recent)

    def save_state(self) -> dict:
        return {
            'log_var': self.log_var.tolist(),
            'mean': self.mean.tolist(),
        }

    def load_state(self, state: dict) -> None:
        self.log_var = torch.tensor(state['log_var'], device=self.device)
        self.mean = torch.tensor(state['mean'], device=self.device)


class ActiveInferenceModule:
    """主动推理模块 — FEP 动作选择

    期望自由能 G = -info_gain - pragmatic_value + risk_penalty
    最小化 G = 最大化信息增益 + 工具价值 - 最小化风险
    """

    def __init__(self, action_dim: int, obs_dim: int = 40):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.preference_mu: Optional[torch.Tensor] = None
        self.preference_precision: float = 1.0
        self.risk_penalty_weight: float = 0.0
        self.gef_history: deque = deque(maxlen=100)
        self.risk_history: deque = deque(maxlen=100)

    def set_preference(self, target_state: torch.Tensor,
                       precision: float = 1.0) -> None:
        self.preference_mu = target_state.detach().clone()
        self.preference_precision = precision

    def set_risk_sensitivity(self, weight: float) -> None:
        self.risk_penalty_weight = weight

    def compute_expected_free_energy(
        self, obs: torch.Tensor, action_idx: int,
        predictor: ProbabilisticPredictor,
        belief: VariationalBelief,
    ) -> float:
        action = torch.zeros(self.action_dim, device=obs.device)
        action[action_idx] = 1.0
        obs_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
        act_in = action.unsqueeze(0)

        mean, log_var = predictor(obs_in, act_in)
        mean = mean.squeeze(0)
        log_var = log_var.squeeze(0)
        var = torch.exp(log_var)

        current_uncertainty = torch.exp(belief.log_var)
        info_gain = torch.sum(
            torch.log(current_uncertainty + 1e-8)
            - torch.log(var + 1e-8)
        )

        pragmatic_value = torch.tensor(0.0, device=obs.device)
        if self.preference_mu is not None:
            pref = self.preference_mu[:mean.shape[0]].to(obs.device)
            pref_error = mean - pref
            pragmatic_value = -0.5 * self.preference_precision * torch.sum(pref_error ** 2)

        risk_term = torch.tensor(0.0, device=obs.device)
        if self.risk_penalty_weight > 0:
            belief_precision = torch.exp(-belief.log_var)
            risk_term = torch.sum(var * belief_precision)

        gef = -info_gain - pragmatic_value + self.risk_penalty_weight * risk_term
        return gef.item()

    def select_action(
        self, obs: torch.Tensor, predictor: ProbabilisticPredictor,
        belief: VariationalBelief, available_actions: List[int],
        noise_scale: float = 0.1,
    ) -> int:
        best_action = available_actions[0]
        best_gef = float('inf')
        best_risk = 0.0

        for action in available_actions:
            gef = self.compute_expected_free_energy(obs, action, predictor, belief)
            noise = torch.randn(1).item() * noise_scale
            gef_noisy = gef + noise

            if gef_noisy < best_gef:
                best_gef = gef_noisy
                best_action = action
                best_risk = 0.0
                if self.risk_penalty_weight > 0:
                    act_t = torch.zeros(self.action_dim, device=obs.device)
                    act_t[action] = 1.0
                    _, lv = predictor(
                        obs.unsqueeze(0) if obs.dim() == 1 else obs,
                        act_t.unsqueeze(0),
                    )
                    var = torch.exp(lv.squeeze(0))
                    bp = torch.exp(-belief.log_var)
                    best_risk = torch.sum(var * bp).item()

        self.gef_history.append(best_gef)
        if self.risk_penalty_weight > 0:
            self.risk_history.append(best_risk)
        return best_action

    def select_action_batch(
        self, obs: torch.Tensor, predictor: ProbabilisticPredictor,
        belief: VariationalBelief, available_actions: List[int],
        noise_scale: float = 0.1,
    ) -> int:
        n = len(available_actions)
        if n == 1:
            return available_actions[0]

        device = obs.device
        obs_in = obs.unsqueeze(0).expand(n, -1)
        act_batch = torch.zeros(n, self.action_dim, device=device)
        for i, a in enumerate(available_actions):
            act_batch[i, a] = 1.0

        mean_batch, logvar_batch = predictor(obs_in, act_batch)
        var_batch = torch.exp(logvar_batch)

        current_unc = torch.exp(belief.log_var)
        info_gain = torch.sum(
            torch.log(current_unc + 1e-8).unsqueeze(0)
            - torch.log(var_batch + 1e-8),
            dim=1,
        )

        pragmatic = torch.zeros(n, device=device)
        if self.preference_mu is not None:
            pref = self.preference_mu[:mean_batch.shape[1]].to(device)
            pragmatic = -0.5 * self.preference_precision * torch.sum(
                (mean_batch - pref) ** 2, dim=1
            )

        risk = torch.zeros(n, device=device)
        if self.risk_penalty_weight > 0:
            bp = torch.exp(-belief.log_var)
            risk = torch.sum(var_batch * bp, dim=1)

        gef = -info_gain - pragmatic + self.risk_penalty_weight * risk
        gef_noisy = gef + torch.randn(n, device=device) * noise_scale

        best_idx = torch.argmin(gef_noisy).item()
        best_action = available_actions[best_idx]

        self.gef_history.append(gef_noisy[best_idx].item())
        if self.risk_penalty_weight > 0:
            self.risk_history.append(risk[best_idx].item())
        return best_action

    def get_avg_gef(self) -> float:
        if not self.gef_history:
            return 0.0
        return sum(self.gef_history) / len(self.gef_history)

    def get_exploration_drive(self) -> float:
        if len(self.gef_history) < 10:
            return 1.0
        recent = list(self.gef_history)[-10:]
        t = torch.tensor(recent)
        return min(1.0, t.var().item() / (t.abs().mean().item() + 1e-8))

    def get_risk_drive(self) -> float:
        if not self.risk_history:
            return 0.0
        return sum(self.risk_history) / len(self.risk_history)

    def save_state(self) -> dict:
        return {
            'preference_mu': self.preference_mu.tolist() if self.preference_mu is not None else None,
            'preference_precision': self.preference_precision,
            'risk_penalty_weight': self.risk_penalty_weight,
            'gef_history': list(self.gef_history),
            'risk_history': list(self.risk_history),
        }

    def load_state(self, state: dict) -> None:
        if state.get('preference_mu') is not None:
            self.preference_mu = torch.tensor(state['preference_mu'])
        self.preference_precision = state.get('preference_precision', 1.0)
        self.risk_penalty_weight = state.get('risk_penalty_weight', 0.0)
        self.gef_history = deque(state.get('gef_history', []), maxlen=100)
        self.risk_history = deque(state.get('risk_history', []), maxlen=100)
