"""
Optimization solver for VFEEL Problem P1.

Implements Algorithm 1 and Algorithm 2 from the paper:
- Algorithm 1: Alternating optimization for (p_k, η) given (p_s, b)
- Algorithm 2: Joint optimization of all variables
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional

from .model import SystemConfig, VFEELOptimizer, AirCompModel, VFEELConvergenceBound


class VFEELSolver:
    """
    Full solver implementing the paper's optimization framework.

    Supports:
    - Grid search (baseline)
    - Alternating optimization (Algorithm 2)
    - Convex optimization for subproblems (Proposition 1)
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.K = config.K
        self.aircomp = AirCompModel(config)
        self.convergence = VFEELConvergenceBound(config)
        self.optimizer = VFEELOptimizer(config)

    def solve_grid_search(
        self,
        E_budget: float,
        T_budget: float,
        p_s_values: Optional[np.ndarray] = None,
        b_values: Optional[np.ndarray] = None,
        P_t_values: Optional[np.ndarray] = None,
        eta_values: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Grid search solver (baseline comparison).

        Searches over discrete values of (p_s, b, p_k, η).
        """
        if p_s_values is None:
            p_s_values = np.linspace(0.01, self.config.P_s_max, 5)
        if b_values is None:
            b_values = np.array([50, 75, 100, 125, 150, 175, 200])
        if P_t_values is None:
            P_t_values = np.linspace(0.01, self.config.P_t_max, 5)
        if eta_values is None:
            eta_values = np.linspace(0.1, 0.9, 5)

        best_omega = float('inf')
        best_params = {}

        for p_s in p_s_values:
            for b in b_values:
                for P_t in P_t_values:
                    for eta in eta_values:
                        # Compute latency and energy
                        p_k = np.ones(self.K) * P_t / self.K
                        T = self.convergence.latency_per_round(b, p_s, p_k)
                        E = self.convergence.energy_per_round(b, p_s, p_k)

                        if T > T_budget or E > E_budget:
                            continue

                        omega = self.convergence.compute_bound(
                            b, p_s, p_k, eta, self.aircomp
                        )

                        if omega < best_omega:
                            best_omega = omega
                            best_params = {
                                'p_s': p_s,
                                'b': b,
                                'p_k': p_k,
                                'eta': eta,
                                'omega': omega,
                                'latency': T,
                                'energy': E,
                            }

        return best_params

    def solve_alternating(
        self,
        E_budget: float,
        T_budget: float,
        max_iterations: int = 20,
        tol: float = 1e-6,
    ) -> dict:
        """
        Alternating optimization (Algorithm 2 from paper).

        Returns:
            Dictionary with optimal parameters and convergence history.
        """
        return self.optimizer.solve(E_budget, T_budget, max_iterations)

    def solve_convex_subproblem_p_s_b(
        self,
        E_budget: float,
        T_budget: float,
        p_k: np.ndarray,
        eta: float,
    ) -> tuple[float, int]:
        """
        Solve convex subproblem for p_s and b (Proposition 1).

        Uses scipy.minimize with constraints.
        """
        def objective(x):
            p_s, log_b = x[0], x[1]
            b = int(np.exp(log_b))
            if b < 1:
                b = 1
            omega = self.convergence.compute_bound(
                b, p_s, p_k, eta, self.aircomp
            )
            return omega

        def constraint_latency(x):
            p_s, log_b = x[0], x[1]
            b = int(np.exp(log_b))
            b = max(b, 1)
            T = self.convergence.latency_per_round(b, p_s, p_k)
            return T_budget - T

        def constraint_energy(x):
            p_s, log_b = x[0], x[1]
            b = int(np.exp(log_b))
            b = max(b, 1)
            E = self.convergence.energy_per_round(b, p_s, p_k)
            return E_budget - E

        # Initial guess
        x0 = [self.config.P_s_max * 0.5, np.log(150)]

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=[(1e-6, self.config.P_s_max), (np.log(1), np.log(500))],
            constraints=[
                {'type': 'ineq', 'fun': constraint_latency},
                {'type': 'ineq', 'fun': constraint_energy},
            ],
            options={'maxiter': 100, 'ftol': 1e-9},
        )

        p_s_opt = float(result.x[0])
        b_opt = int(np.exp(result.x[1]))
        b_opt = max(b_opt, 1)

        return p_s_opt, b_opt

    def solve_convex_subproblem_p_k_eta(
        self,
        batch_size: int,
        sensing_power: float,
    ) -> tuple[np.ndarray, float]:
        """
        Solve convex subproblem for p_k and η (Algorithm 1).

        Alternating between:
        - Optimal η (Eq.44): η* = σ²_n / (σ²_n + Σ h_k² · p_k)
        - Optimal p_k (Eq.50): water-filling with constraints
        """
        return self.optimizer.optimize_step2_power_denoising(
            batch_size, sensing_power
        )

    def get_optimal_eta_closed_form(
        self,
        transmit_powers: np.ndarray,
        channel_norms: Optional[np.ndarray] = None,
    ) -> float:
        """
        Optimal η closed-form (Eq.44 in paper).

        η* = σ²_n / (σ²_n + Σ h_k² · p_k)
        """
        sigma2_n = self.config.sigma2_n_linear
        if channel_norms is None:
            channel_norms = np.abs(self.aircomp.channels) ** 2

        signal_power = np.sum(channel_norms * transmit_powers)
        eta = sigma2_n / (sigma2_n + signal_power + 1e-10)
        return float(np.clip(eta, 0.01, 0.99))

    def get_optimal_power_waterfilling(
        self,
        eta: float,
        P_t_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Optimal transmit power allocation (Eq.50 in paper).

        Water-filling based on channel gains and denoising factor.
        """
        if P_t_max is None:
            P_t_max = self.config.P_t_max

        channels = self.aircomp.channels
        channel_norms = np.abs(channels) ** 2
        K = self.K

        # Water level (simplified)
        # p_k* ∝ |h_k|² · η / σ²_n (high-SNR approximation)
        water_level = 0.5 * P_t_max  # start mid-range
        powers = water_level * channel_norms / np.mean(channel_norms)

        # Normalize to total power constraint
        total_power = np.sum(powers)
        if total_power > K * P_t_max:
            powers = powers * (K * P_t_max) / total_power

        powers = np.clip(powers, 1e-6, P_t_max)
        return powers
