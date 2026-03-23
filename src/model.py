"""
System and signal models for VFEEL (Vertical Federated Edge Learning).

Paper: "Joint Sensing, Communication, and Computation for Vertical Federated
Edge Learning in Edge Perception Networks" (arXiv:2512.03374)

Key models:
- K edge devices + 1 edge server
- ISAC transceiver (time-division mode)
- Sensing noise model (Eq.6)
- AirComp aggregation model (Eq.11)
- Convergence bound Ω (based on gradient variance and aggregation error)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for system configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceConfig:
    """Configuration for a single edge device."""
    k: int                    # device index (0 to K-1)
    p_s: float = 1.0         # sensing power
    p_t: float = 1.0         # transmission power
    batch_size: int = 150    # local batch size
    eta: float = 0.5         # denoising factor (0 < eta <= 1)
    h_k: complex = 1.0 + 0j   # channel coefficient


@dataclass
class SystemConfig:
    """Full VFEEL system configuration."""
    K: int = 3                # number of edge devices
    d_model: int = 512        # model parameter dimension
    T_max: float = 100e-3     # max latency per round (seconds)
    E_max: float = 1.0        # max energy budget per round (Joules)
    P_s_max: float = 1.0      # max sensing power
    P_t_max: float = 1.0     # max transmit power per device
    sigma2_s: float = 1e-6   # sensing noise power spectral density
    sigma2_n: float = -100   # noise power in dBm (converted to linear)
    gamma_s: float = 1e-3    # sensing SNR penalty factor
    f_c: float = 77e9        # carrier frequency (Hz) for ISAC
    B: float = 100e6          # bandwidth (Hz)
    tau_s: float = 0.3        # sensing time fraction (0 < tau_s < 1)
    # Learning hyperparameters
    lr: float = 0.01          # learning rate
    momentum: float = 0.9     # SGD momentum
    num_classes: int = 7      # human motion recognition (7 classes)
    # ResNet-10 proxy: total params ~ 2.5M, we use a reduced model for sim
    model_param_count: int = int(2.5e6)
    # Communication
    beta: float = 0.5         # AirComp scaling factor

    def __post_init__(self):
        # Convert dBm to linear Watt
        self.sigma2_n_linear = 1e-3 * 10 ** (self.sigma2_n / 10)


# ─────────────────────────────────────────────────────────────────────────────
# Core model components
# ─────────────────────────────────────────────────────────────────────────────

class SensingNoiseModel:
    """
    Sensing noise model (Eq.6 in paper):
    ξ̃ = ξ + γ_s + n_s/√P_s

    The effective sensing noise includes:
    - ξ: hardware impairment
    - γ_s: clutter noise
    - n_s/√P_s: normalized thermal noise
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.sigma2_s = config.sigma2_s
        self.gamma_s = config.gamma_s
        self.sigma2_n = config.sigma2_n_linear

    def effective_noise_variance(self, p_s: float) -> float:
        """
        Compute effective sensing noise variance for given sensing power.

        Args:
            p_s: Sensing power (linear)

        Returns:
            Effective noise variance
        """
        # ξ̃ = ξ + γ_s + n_s/√P_s
        # Variance: E[|ξ̃|²] = σ²_ξ + γ_s + σ²_n/P_s
        # We model E[|ξ|²] ≈ σ²_s as the hardware impairment term
        return self.sigma2_s + self.gamma_s + self.sigma2_n / p_s


class AirCompModel:
    """
    AirComp aggregation model (Eq.11 in paper):
    y_i = Σ_k h_k · √p_k · ψ_k + z

    Models the over-the-air computation for federated averaging
    with ISAC transmission.
    """

    def __init__(self, config: SystemConfig, channels: Optional[np.ndarray] = None):
        self.config = config
        # Channel coefficients: h_k for each device
        if channels is None:
            # Default: i.i.d. Rayleigh fading
            self.channels = np.sqrt(0.5) * (
                np.random.randn(config.K) + 1j * np.random.randn(config.K)
            )
        else:
            self.channels = channels

    def aggregation_signal(self, psi: np.ndarray, powers: np.ndarray) -> np.ndarray:
        """
        Compute AirComp aggregated signal.

        Args:
            psi: Local updates from each device [K, d]
            powers: Transmission powers [K]

        Returns:
            Aggregated signal at server
        """
        K = self.config.K
        y = np.zeros(psi.shape[1], dtype=np.complex128)
        for k in range(K):
            y += self.channels[k] * np.sqrt(powers[k]) * psi[k]
        # Add AWGN at receiver
        z = np.sqrt(0.5 * self.config.sigma2_n_linear) * (
            np.random.randn(psi.shape[1]) + 1j * np.random.randn(psi.shape[1])
        )
        y += z
        return y

    def aggregation_mse(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        channel_norms: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute aggregation MSE (Eq.12-Eq.15 proxy).

        The MSE depends on:
        - batch size (gradient variance ∝ 1/b)
        - sensing power (affects local gradient quality)
        - transmit powers (signal strength)
        - denoising factor η (controls noise suppression)
        - channel conditions

        Returns:
            MSE of the aggregated gradient estimate
        """
        K = self.config.K
        sigma2_n = self.config.sigma2_n_linear

        if channel_norms is None:
            channel_norms = np.abs(self.channels) ** 2  # [K]

        # Signal power from each device
        signal_power = channel_norms * transmit_powers  # [K]

        # Total received signal power
        P_total = np.sum(signal_power)

        # Noise at aggregation: scaled by batch_size (larger batch = less noise)
        # Gradient variance scales as σ²/b per device
        # Effective noise = σ²_n / (b * P_s) for sensing + σ²_n for comm
        sensing_model = SensingNoiseModel(self.config)
        sensing_noise_var = sensing_model.effective_noise_variance(sensing_power)

        # Aggregation error variance
        # The AirComp MSE formula (approximation from paper Eq.14-Eq.15):
        # MSE = (σ²_sensing / b + σ²_comm) / (Σ√p_k h_k)²
        # With denoising factor η: MSE scales as (1-η) component + noise/b
        noise_component = sensing_noise_var / batch_size + sigma2_n / np.sum(signal_power)

        # Denoising factor effect: η reduces noise but may lose signal
        # Effective MSE = noise_component / η² (approximation)
        mse = noise_component / (denoising_factor ** 2 + 1e-10)
        return float(np.real(mse))


class VFEELConvergenceBound:
    """
    Convergence bound Ω for VFEEL (Problem P1 objective).

    The bound captures the effect of:
    - Local gradient variance (∝ 1/batch_size)
    - Aggregation error (from sensing + AirCOMP)
    - Communication latency
    - Energy consumption
    """

    def __init__(self, config: SystemConfig):
        self.config = config

    def compute_bound(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        aircomp_model: AirCompModel,
    ) -> float:
        """
        Compute the convergence upper bound Ω (Eq.16-Eq.18 in paper).

        Args:
            batch_size: Local batch size per device
            sensing_power: Sensing power
            transmit_powers: Transmission power per device [K]
            denoising_factor: Denoising factor η
            aircomp_model: AirCOMP model instance

        Returns:
            Convergence bound Ω (lower is better)
        """
        K = self.config.K

        # Aggregation MSE
        agg_mse = aircomp_model.aggregation_mse(
            batch_size, sensing_power, transmit_powers, denoising_factor
        )

        # Local gradient variance: ∝ 1/batch_size
        # For SGD: Var[∇F] ≈ σ²/b where σ² is data variance
        gradient_variance = 1.0 / batch_size

        # Convergence bound (Eq.18 proxy):
        # Ω = (1 - 2η)/2μ · [σ²_g/b + K·MSE_agg + ...]
        # where μ is the strong concavity parameter
        mu = 0.01  # strong concavity (typical for ML objectives)

        # Bound formula from paper (simplified):
        # Ω = (σ²_g / b + K * MSE_agg) / (2 * μ)
        omega = (gradient_variance / batch_size + K * agg_mse) / (2 * mu)

        return float(np.real(omega))

    def latency_per_round(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
    ) -> float:
        """
        Compute per-round latency (Eq.3-Eq.5 proxy).

        Components:
        - Sensing time: τ_s (depends on sensing_power and channel)
        - Local computation time: b / f_k (CPU frequency)
        - Transmission time: model_size / (B * log_SNR)
        - Aggregation time: negligible
        """
        K = self.config.K
        B = self.config.B
        tau_s = self.config.tau_s

        # Computation latency (CPU cycles per sample × batch / CPU freq)
        # Approximate: 10K cycles/sample for ResNet-10 forward+backward
        cycles_per_sample = 1e4
        cpu_freq = 2e9  # 2 GHz
        T_comp = (batch_size * cycles_per_sample) / cpu_freq  # seconds

        # Communication latency (AirCOMP)
        # Model size ~ 2.5M params × 4 bytes = 10 MB
        model_size_bytes = self.config.model_param_count * 4
        # Effective rate: B * log2(1+SNR) with SNR = P_t * |h|² / σ²_n
        avg_snr = np.mean(transmit_powers) / (self.config.sigma2_n_linear + 1e-10)
        rate_bps = B * np.log2(1 + avg_snr)  # bits/second
        T_comm = model_size_bytes * 8 / rate_bps  # seconds

        # Sensing time (fixed fraction)
        T_sense = tau_s * self.config.T_max

        # Total latency
        T_total = T_sense + T_comp + T_comm

        return T_total

    def energy_per_round(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
    ) -> float:
        """
        Compute per-round energy consumption (Eq.1-Eq.2 proxy).

        Components:
        - Sensing energy: P_s × τ_s
        - Computation energy: α × b (α = energy per sample)
        - Transmission energy: Σ P_k × T_comm
        """
        K = self.config.K
        tau_s = self.config.tau_s

        # Sensing energy
        E_sense = sensing_power * tau_s * 1e-3  # mW × s → J

        # Computation energy: ~10 pJ/sample for modern CPUs
        alpha = 10e-12  # J/sample
        E_comp = batch_size * K * alpha

        # Transmission energy
        # T_comm approximation (see latency_per_round)
        model_size_bytes = self.config.model_param_count * 4
        avg_snr = np.mean(transmit_powers) / (self.config.sigma2_n_linear + 1e-10)
        B = self.config.B
        rate_bps = B * np.log2(1 + avg_snr)
        T_comm = model_size_bytes * 8 / rate_bps
        E_comm = np.sum(transmit_powers) * T_comm

        E_total = E_sense + E_comp + E_comm
        return E_total


class VFEELOptimizer:
    """
    Alternating optimization solver for Problem P1.

    Algorithm 2 from paper:
    1. Optimize sensing power p_s and batch size b (convex, Proposition 1)
    2. Optimize transmit powers p_k and denoising factor η (Algorithm 1)
       - Optimal η: Eq.44
       - Optimal p_k: Eq.50
    3. Iterate until convergence
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.aircomp = AirCompModel(config)
        self.convergence = VFEELConvergenceBound(config)

    def optimize_step1_batch_sensing(
        self,
        E_budget: float,
        T_budget: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
    ) -> tuple[float, int]:
        """
        Step 1: Optimize sensing power p_s and batch size b.

        Uses convex optimization (Proposition 1 in paper).
        The optimal b and p_s trade off latency/energy constraints.

        Returns:
            (optimal_sensing_power, optimal_batch_size)
        """
        K = self.config.K
        P_s_max = self.config.P_s_max

        # Envelope for search (p_s, b)
        best_p_s = P_s_max / 2
        best_b = 150  # default from paper

        best_obj = float('inf')

        for p_s in np.linspace(0.01 * P_s_max, P_s_max, 20):
            for b in [50, 75, 100, 125, 150, 175, 200, 250]:
                # Check constraints
                T = self.convergence.latency_per_round(b, p_s, transmit_powers)
                E = self.convergence.energy_per_round(b, p_s, transmit_powers)

                if T <= T_budget and E <= E_budget:
                    omega = self.convergence.compute_bound(
                        b, p_s, transmit_powers, denoising_factor, self.aircomp
                    )
                    if omega < best_obj:
                        best_obj = omega
                        best_p_s = p_s
                        best_b = b

        return best_p_s, best_b

    def optimize_step2_power_denoising(
        self,
        batch_size: int,
        sensing_power: float,
    ) -> tuple[np.ndarray, float]:
        """
        Step 2: Optimize transmit powers p_k and denoising factor η.

        Algorithm 1 (alternating optimization):
        - Optimal η: Eq.44
        - Optimal p_k: Eq.50

        Returns:
            (optimal_transmit_powers [K], optimal_denoising_factor)
        """
        K = self.config.K
        P_t_max = self.config.P_t_max
        sigma2_n = self.config.sigma2_n_linear
        channels = self.aircomp.channels

        # Compute optimal η (Eq.44)
        # η* = σ²_n / (σ²_n + Σ h_k² · p_k)  [simplified from paper]
        # We iterate over η as part of alternating optimization
        def compute_optimal_eta(powers: np.ndarray) -> float:
            signal_power = np.sum(np.abs(channels) ** 2 * powers)
            eta = sigma2_n / (sigma2_n + signal_power + 1e-10)
            return np.clip(eta, 0.01, 0.99)

        # Compute optimal p_k (Eq.50) - water-filling like solution
        def compute_optimal_powers(eta: float) -> np.ndarray:
            # Simplified: equal power allocation with channel weighting
            # p_k* ∝ |h_k|² / η (from Eq.50 simplified)
            channel_weights = np.abs(channels) ** 2
            # Normalize to satisfy sum constraint
            total_power = K * P_t_max * 0.5  # start with 50% of max
            powers = (channel_weights / np.sum(channel_weights)) * total_power
            powers = np.clip(powers, 0.01 * P_t_max, P_t_max)
            return powers

        # Alternating optimization (Algorithm 1)
        powers = np.ones(K) * P_t_max * 0.5
        eta = 0.5

        for _ in range(10):
            eta_new = compute_optimal_eta(powers)
            powers_new = compute_optimal_powers(eta_new)

            if np.abs(eta_new - eta) < 1e-4 and np.max(np.abs(powers_new - powers)) < 1e-4:
                break
            eta = eta_new
            powers = powers_new

        return powers, float(eta)

    def solve(
        self,
        E_budget: float,
        T_budget: float,
        max_iterations: int = 20,
    ) -> dict:
        """
        Main solver: Algorithm 2 from paper.

        Alternating optimization of (p_s, b) and (p_k, η).

        Returns:
            Dictionary with optimal values and convergence history
        """
        K = self.config.K
        P_t_max = self.config.P_t_max

        # Initialize
        p_s = self.config.P_s_max * 0.5
        b = 150
        p_k = np.ones(K) * P_t_max * 0.5
        eta = 0.5

        history = []

        for iteration in range(max_iterations):
            # Step 1: Optimize (p_s, b) with fixed (p_k, η)
            p_s_new, b_new = self.optimize_step1_batch_sensing(
                E_budget, T_budget, p_k, eta
            )

            # Step 2: Optimize (p_k, η) with fixed (p_s, b)
            p_k_new, eta_new = self.optimize_step2_power_denoising(b_new, p_s_new)

            # Check convergence
            omega = self.convergence.compute_bound(
                b_new, p_s_new, p_k_new, eta_new, self.aircomp
            )

            history.append({
                'iteration': iteration,
                'p_s': p_s_new,
                'b': b_new,
                'p_k': p_k_new,
                'eta': eta_new,
                'omega': omega,
            })

            p_s, b, p_k, eta = p_s_new, b_new, p_k_new, eta_new

        return {
            'optimal_p_s': p_s,
            'optimal_b': b,
            'optimal_p_k': p_k,
            'optimal_eta': eta,
            'final_omega': omega,
            'history': history,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FL simulation proxy (simplified ResNet-10 / 7-class recognition)
# ─────────────────────────────────────────────────────────────────────────────

class VFEELSimulator:
    """
    Simplified VFEEL simulator for reproducing figures.

    Models:
    - K devices training on human motion recognition
    - ISAC-based AirCOMP aggregation
    - Convergence driven by the optimization framework
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.K = config.K
        self.aircomp = AirCompModel(config)
        self.convergence = VFEELConvergenceBound(config)
        self.optimizer = VFEELOptimizer(config)

    def simulate_round(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        current_accuracy: float,
    ) -> float:
        """
        Simulate one VFEEL round.

        Returns:
            New accuracy after one round
        """
        # Compute aggregation MSE
        mse = self.aircomp.aggregation_mse(
            batch_size, sensing_power, transmit_powers, denoising_factor
        )

        # Convergence rate: accuracy improvement inversely proportional to MSE
        # Base improvement per round (typical for ResNet-10 on 7-class)
        base_improvement = 0.02

        # MSE penalty on convergence
        # Higher MSE → slower convergence
        mse_penalty = np.log1p(mse * 1e6) / 100  # scaled penalty
        improvement = base_improvement / (1 + mse_penalty)

        new_accuracy = min(current_accuracy + improvement, 0.95)

        # Add small noise to simulate stochasticity
        noise = np.random.randn() * 0.005
        new_accuracy = np.clip(new_accuracy + noise, 0.0, 0.95)

        return float(new_accuracy)

    def simulate_convergence_curve(
        self,
        batch_size: int,
        sensing_power: float,
        transmit_powers: np.ndarray,
        denoising_factor: float,
        num_rounds: int = 100,
        initial_accuracy: float = 0.15,
    ) -> np.ndarray:
        """
        Simulate accuracy vs communication rounds.

        Reproduces Fig.4 (batch size vs convergence).
        """
        accuracies = [initial_accuracy]
        acc = initial_accuracy

        for r in range(num_rounds):
            acc = self.simulate_round(
                batch_size, sensing_power, transmit_powers, denoising_factor, acc
            )
            accuracies.append(acc)

        return np.array(accuracies)

    def reproduce_figure_4(
        self,
        batch_sizes: list[int] = [50, 100, 150, 200],
        num_rounds: int = 100,
    ) -> dict:
        """
        Reproduce Fig.4: Impact of batch size on convergence.

        Curves: Test accuracy vs communication rounds for different batch sizes.
        Expected: Larger batch → faster initial convergence, similar final accuracy.
        """
        K = self.config.K
        P_t_max = self.config.P_t_max
        results = {}

        # Fixed sensing power and transmission
        p_s = 0.5  # mid-range
        p_k = np.ones(K) * P_t_max * 0.3
        eta = 0.5

        for b in batch_sizes:
            acc_curve = self.simulate_convergence_curve(
                b, p_s, p_k, eta, num_rounds
            )
            results[f'b={b}'] = acc_curve

        return results

    def reproduce_figure_5(
        self,
        noise_powers_dbm: list[float] = [-110, -100, -90, -80],
        num_rounds: int = 100,
    ) -> dict:
        """
        Reproduce Fig.5: Impact of channel noise on performance.

        Noise power varies (in dBm). Higher noise → worse performance.
        """
        results = {}
        K = self.config.K
        P_t_max = self.config.P_t_max

        b = 150  # optimal batch size from paper
        p_s = 0.5
        p_k = np.ones(K) * P_t_max * 0.3

        for noise_dbm in noise_powers_dbm:
            # Update noise level
            self.config.sigma2_n = noise_dbm
            self.config.sigma2_n_linear = 1e-3 * 10 ** (noise_dbm / 10)
            # Re-create models with new noise
            self.aircomp = AirCompModel(self.config)
            self.convergence = VFEELConvergenceBound(self.config)

            eta = 0.5  # keep fixed for comparison
            acc_curve = self.simulate_convergence_curve(
                b, p_s, p_k, eta, num_rounds
            )
            results[f'σ²_n={noise_dbm}dBm'] = acc_curve

        return results

    def reproduce_figure_6(
        self,
        latency_constraints_ms: list[float] = [50, 100, 150, 200],
        num_rounds: int = 100,
    ) -> dict:
        """
        Reproduce Fig.6: Impact of latency constraint on test accuracy.

        Solve optimization for each latency budget, simulate convergence.
        """
        results = {}
        K = self.config.K
        P_t_max = self.config.P_t_max

        E_budget = self.config.E_max
        # Re-create optimizer
        self.optimizer = VFEELOptimizer(self.config)

        for T_max_ms in latency_constraints_ms:
            T_budget = T_max_ms * 1e-3  # convert to seconds

            # Solve optimization
            opt_result = self.optimizer.solve(E_budget, T_budget)

            # Simulate with optimal parameters
            acc_curve = self.simulate_convergence_curve(
                opt_result['optimal_b'],
                opt_result['optimal_p_s'],
                opt_result['optimal_p_k'],
                opt_result['optimal_eta'],
                num_rounds,
            )
            results[f'T_max={T_max_ms}ms'] = {
                'curve': acc_curve,
                'optimal_b': opt_result['optimal_b'],
                'optimal_p_s': opt_result['optimal_p_s'],
                'optimal_eta': opt_result['optimal_eta'],
            }

        return results

    def reproduce_figure_7(
        self,
        energy_budgets: list[float] = [0.5, 1.0, 2.0, 5.0],
        num_rounds: int = 100,
    ) -> dict:
        """
        Reproduce Fig.7: Impact of energy budget on test accuracy.
        """
        results = {}
        K = self.config.K
        P_t_max = self.config.P_t_max

        T_budget = self.config.T_max
        self.optimizer = VFEELOptimizer(self.config)

        for E_budget in energy_budgets:
            opt_result = self.optimizer.solve(E_budget, T_budget)

            acc_curve = self.simulate_convergence_curve(
                opt_result['optimal_b'],
                opt_result['optimal_p_s'],
                opt_result['optimal_p_k'],
                opt_result['optimal_eta'],
                num_rounds,
            )
            results[f'E={E_budget}J'] = {
                'curve': acc_curve,
                'optimal_b': opt_result['optimal_b'],
                'optimal_p_s': opt_result['optimal_p_s'],
                'optimal_eta': opt_result['optimal_eta'],
            }

        return results

    def reproduce_figure_8(
        self,
        max_power_values: list[float] = [0.1, 0.5, 1.0, 2.0],
        num_rounds: int = 100,
    ) -> dict:
        """
        Reproduce Fig.8: Impact of max transmit power on performance.
        """
        results = {}
        K = self.config.K

        b = 150
        p_s = 0.5
        eta = 0.5

        for P_t_max in max_power_values:
            self.config.P_t_max = P_t_max
            p_k = np.ones(K) * P_t_max * 0.3

            acc_curve = self.simulate_convergence_curve(
                b, p_s, p_k, eta, num_rounds
            )
            results[f'P_t_max={P_t_max}W'] = acc_curve

        return results
