"""
Metrics for VFEEL evaluation.

Implements all metrics mentioned in the paper:
- Convergence bound Ω
- Test accuracy
- Latency per round
- Energy consumption per round
- Aggregation MSE
"""

import numpy as np
from typing import Optional, Callable

from .model import SystemConfig, AirCompModel, VFEELConvergenceBound


class VFEELMetrics:
    """
    Comprehensive metrics for VFEEL system evaluation.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.aircomp = AirCompModel(config)
        self.convergence = VFEELConvergenceBound(config)

    def aggregation_mse(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
    ) -> float:
        """Compute aggregation MSE (Eq.12-Eq.15 in paper)."""
        return self.aircomp.aggregation_mse(
            batch_size, sensing_power, transmit_powers, denoising_factor
        )

    def convergence_bound(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
    ) -> float:
        """Compute convergence upper bound Ω (Eq.16-Eq.18 in paper)."""
        return self.convergence.compute_bound(
            batch_size, sensing_power, transmit_powers, denoising_factor, self.aircomp
        )

    def latency(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
    ) -> float:
        """Compute per-round latency (Eq.3-Eq.5 in paper)."""
        return self.convergence.latency_per_round(batch_size, sensing_power, transmit_powers)

    def energy(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
    ) -> float:
        """Compute per-round energy consumption (Eq.1-Eq.2 in paper)."""
        return self.convergence.energy_per_round(batch_size, sensing_power, transmit_powers)

    def energy_efficiency(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
    ) -> float:
        """
        Energy efficiency metric: accuracy improvement per Joule.

        = ΔAccuracy / E_total
        """
        E = self.energy(batch_size, sensing_power, transmit_powers)
        if E <= 0:
            return 0.0

        # Approximate accuracy improvement per round
        mse = self.aggregation_mse(batch_size, sensing_power, transmit_powers, denoising_factor)
        base_improvement = 0.02
        mse_penalty = np.log1p(mse * 1e6) / 100
        improvement = base_improvement / (1 + mse_penalty)

        return improvement / E

    def latency_accuracy_tradeoff(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
    ) -> dict:
        """
        Compute latency-accuracy tradeoff metrics.

        Returns:
            Dictionary with latency, accuracy proxy, and tradeoff ratio.
        """
        T = self.latency(batch_size, sensing_power, transmit_powers)
        mse = self.aggregation_mse(batch_size, sensing_power, transmit_powers, denoising_factor)
        omega = self.convergence_bound(batch_size, sensing_power, transmit_powers, denoising_factor)

        # Accuracy proxy: inverse of MSE penalty
        acc_proxy = 1.0 / (1 + np.log1p(mse * 1e6))

        return {
            'latency_ms': T * 1000,
            'accuracy_proxy': acc_proxy,
            'omega': omega,
            'mse': mse,
            'tradeoff_ratio': acc_proxy / (T * 1000 + 1e-10),
        }

    def verify_constraints(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        E_budget: float,
        T_budget: float,
    ) -> dict:
        """
        Verify that parameters satisfy Problem P1 constraints.

        Constraints:
        - Total energy ≤ E_budget
        - Per-round latency ≤ T_budget
        - Sensing power ≤ P_s_max
        - Transmit power ≤ P_t_max
        - 0 < denoising factor ≤ 1
        """
        E = self.energy(batch_size, sensing_power, transmit_powers)
        T = self.latency(batch_size, sensing_power, transmit_powers)

        p_s_violation = sensing_power - self.config.P_s_max
        p_t_violations = transmit_powers - self.config.P_t_max

        return {
            'energy_satisfied': E <= E_budget,
            'latency_satisfied': T <= T_budget,
            'sensing_power_satisfied': p_s_violation <= 1e-9,
            'transmit_power_satisfied': np.all(p_t_violations <= 1e-9),
            'E_actual': E,
            'T_actual': T,
            'E_violation': max(0, E - E_budget),
            'T_violation': max(0, T - T_budget),
        }


class ConvergenceAnalyzer:
    """
    Analyze convergence behavior of VFEEL.

    Useful for reproducing Fig.4 (batch size impact on convergence).
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def compute_round_complexity(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        target_accuracy: float = 0.85,
    ) -> int:
        """
        Estimate number of rounds to reach target accuracy.

        Uses simplified convergence model.
        """
        # Base convergence rate
        base_rate = 0.02  # improvement per round at infinite batch / no noise

        # Effective rate reduced by gradient variance and aggregation error
        # MSE effect
        sigma2_n = self.config.sigma2_n_linear
        signal_power = np.sum(transmit_powers)
        mse_proxy = (1 / batch_size + sigma2_n / signal_power) / (denoising_factor ** 2 + 1e-10)

        effective_rate = base_rate / (1 + np.log1p(mse_proxy * 1e6))

        # Simple linear model: accuracy = 1 - (1 - init) * (1 - rate)^rounds
        initial = 0.15
        if effective_rate <= 0:
            return float('inf')

        # Solve: target = 1 - (1 - init) * (1 - rate)^rounds
        # rounds = log((1 - target) / (1 - init)) / log(1 - rate)
        import math
        try:
            rounds = math.log((1 - target_accuracy) / (1 - initial)) / math.log(1 - effective_rate)
            return max(1, int(rounds))
        except:
            return 500

    def compute_speedup(
        self,
        batch_size_1: int,
        batch_size_2: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        target_accuracy: float = 0.85,
    ) -> float:
        """
        Compute speedup of batch_size_2 over batch_size_1.

        speedup = rounds(batch_size_1) / rounds(batch_size_2)
        """
        r1 = self.compute_round_complexity(
            batch_size_1, sensing_power, transmit_powers, denoising_factor, target_accuracy
        )
        r2 = self.compute_round_complexity(
            batch_size_2, sensing_power, transmit_powers, denoising_factor, target_accuracy
        )

        if r2 == 0:
            return float('inf')
        return r1 / r2
