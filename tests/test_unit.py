"""
Unit tests for VFEEL model components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from src.model import (
    SystemConfig, DeviceConfig,
    SensingNoiseModel, AirCompModel,
    VFEELConvergenceBound, VFEELOptimizer, VFEELSimulator
)
from src.solver import VFEELSolver
from src.metrics import VFEELMetrics, ConvergenceAnalyzer


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return SystemConfig(
        K=3,
        P_s_max=1.0,
        P_t_max=1.0,
        E_max=1.0,
        T_max=100e-3,
        sigma2_n=-100,
        tau_s=0.3,
    )


@pytest.fixture
def sensing_model(default_config):
    return SensingNoiseModel(default_config)


@pytest.fixture
def aircomp_model(default_config):
    return AirCompModel(default_config)


@pytest.fixture
def convergence_model(default_config):
    return VFEELConvergenceBound(default_config)


# ─── SystemConfig ─────────────────────────────────────────────────────────────

def test_system_config_defaults():
    config = SystemConfig()
    assert config.K == 3
    assert config.d_model == 512
    assert config.T_max == 100e-3
    assert config.E_max == 1.0
    assert config.sigma2_n == -100  # dBm
    assert config.sigma2_n_linear == pytest.approx(1e-13, rel=0.1)


def test_system_config_sigma2_conversion():
    config = SystemConfig(sigma2_n=-80)
    # -80 dBm = 1e-3 * 10^(-8) = 1e-11 W
    expected = 1e-3 * 10**(-8)
    assert config.sigma2_n_linear == pytest.approx(expected, rel=0.01)


# ─── SensingNoiseModel ───────────────────────────────────────────────────────

def test_sensing_noise_variance(sensing_model):
    var = sensing_model.effective_noise_variance(p_s=1.0)
    assert var > 0
    assert var < 1e-2  # Should be small for reasonable sensing power


def test_sensing_noise_increases_with_low_power(sensing_model):
    var_high = sensing_model.effective_noise_variance(p_s=1.0)
    var_low = sensing_model.effective_noise_variance(p_s=0.01)
    assert var_low > var_high


# ─── AirCompModel ─────────────────────────────────────────────────────────────

def test_aircomp_channels_shape(aircomp_model, default_config):
    assert len(aircomp_model.channels) == default_config.K


def test_aircomp_aggregation_signal_shape(aircomp_model, default_config):
    d = 64  # parameter dimension
    psi = np.random.randn(default_config.K, d) + 1j * np.random.randn(default_config.K, d)
    powers = np.ones(default_config.K)
    y = aircomp_model.aggregation_signal(psi, powers)
    assert y.shape == (d,)


def test_aircomp_aggregation_mse_positive(aircomp_model, default_config):
    mse = aircomp_model.aggregation_mse(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
    )
    assert mse > 0


def test_aircomp_mse_decreases_with_batch_size(aircomp_model, default_config):
    mse_small = aircomp_model.aggregation_mse(50, 1.0, np.ones(default_config.K) * 0.5, 0.5)
    mse_large = aircomp_model.aggregation_mse(200, 1.0, np.ones(default_config.K) * 0.5, 0.5)
    assert mse_small > mse_large


def test_aircomp_mse_decreases_with_power(aircomp_model, default_config):
    mse_low = aircomp_model.aggregation_mse(150, 1.0, np.ones(default_config.K) * 0.1, 0.5)
    mse_high = aircomp_model.aggregation_mse(150, 1.0, np.ones(default_config.K) * 1.0, 0.5)
    assert mse_low > mse_high


# ─── VFEELConvergenceBound ───────────────────────────────────────────────────

def test_convergence_bound_positive(convergence_model, aircomp_model, default_config):
    omega = convergence_model.compute_bound(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
        aircomp_model=aircomp_model,
    )
    assert omega > 0


def test_convergence_bound_decreases_with_batch(convergence_model, aircomp_model, default_config):
    omega_small = convergence_model.compute_bound(
        50, 1.0, np.ones(default_config.K) * 0.5, 0.5, aircomp_model
    )
    omega_large = convergence_model.compute_bound(
        200, 1.0, np.ones(default_config.K) * 0.5, 0.5, aircomp_model
    )
    assert omega_small > omega_large


def test_latency_positive(convergence_model, default_config):
    T = convergence_model.latency_per_round(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
    )
    assert T > 0
    assert T < 1.0  # Should be < 1 second


def test_energy_positive(convergence_model, default_config):
    E = convergence_model.energy_per_round(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
    )
    assert E > 0


# ─── VFEELOptimizer ────────────────────────────────────────────────────────────

def test_optimizer_step1(default_config):
    optimizer = VFEELOptimizer(default_config)
    p_s, b = optimizer.optimize_step1_batch_sensing(
        E_budget=1.0,
        T_budget=100e-3,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
    )
    assert 0 < p_s <= default_config.P_s_max
    assert b >= 1


def test_optimizer_step2(default_config):
    optimizer = VFEELOptimizer(default_config)
    p_k, eta = optimizer.optimize_step2_power_denoising(
        batch_size=150,
        sensing_power=1.0,
    )
    assert len(p_k) == default_config.K
    assert 0 < eta < 1
    assert np.all(p_k >= 0)


def test_optimizer_solve(default_config):
    optimizer = VFEELOptimizer(default_config)
    result = optimizer.solve(E_budget=1.0, T_budget=100e-3, max_iterations=5)
    assert 'optimal_p_s' in result
    assert 'optimal_b' in result
    assert 'optimal_p_k' in result
    assert 'optimal_eta' in result
    assert 'history' in result


# ─── VFEELSimulator ───────────────────────────────────────────────────────────

def test_simulator_convergence_curve(default_config):
    sim = VFEELSimulator(default_config)
    curve = sim.simulate_convergence_curve(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
        num_rounds=20,
    )
    assert len(curve) == 21
    assert curve[0] == 0.15
    assert np.all(np.diff(curve) >= -0.02)  # Mostly non-decreasing


def test_simulator_fig4_reproduction(default_config):
    sim = VFEELSimulator(default_config)
    results = sim.reproduce_figure_4([50, 100, 150], num_rounds=20)
    assert len(results) == 3
    for key, curve in results.items():
        assert len(curve) == 21


def test_simulator_fig5_reproduction(default_config):
    sim = VFEELSimulator(default_config)
    results = sim.reproduce_figure_5([-100, -90], num_rounds=20)
    assert len(results) == 2


def test_simulator_fig6_reproduction(default_config):
    sim = VFEELSimulator(default_config)
    results = sim.reproduce_figure_6([50, 100], num_rounds=20)
    assert len(results) == 2


def test_simulator_fig8_reproduction(default_config):
    sim = VFEELSimulator(default_config)
    results = sim.reproduce_figure_8([0.5, 1.0], num_rounds=20)
    assert len(results) == 2


# ─── VFEELSolver ──────────────────────────────────────────────────────────────

def test_solver_grid_search(default_config):
    solver = VFEELSolver(default_config)
    result = solver.solve_grid_search(E_budget=1.0, T_budget=100e-3)
    assert 'p_s' in result
    assert 'b' in result
    assert 'omega' in result


def test_solver_alternating(default_config):
    solver = VFEELSolver(default_config)
    result = solver.solve_alternating(E_budget=1.0, T_budget=100e-3, max_iterations=5)
    assert 'optimal_p_s' in result


def test_optimal_eta_formula(default_config):
    solver = VFEELSolver(default_config)
    powers = np.ones(default_config.K) * 0.5
    eta = solver.get_optimal_eta_closed_form(powers)
    assert 0 < eta < 1


def test_waterfilling_power(default_config):
    solver = VFEELSolver(default_config)
    powers = solver.get_optimal_power_waterfilling(eta=0.5)
    assert len(powers) == default_config.K
    assert np.all(powers >= 0)


# ─── VFEELMetrics ─────────────────────────────────────────────────────────────

def test_metrics_aggregation_mse(default_config):
    metrics = VFEELMetrics(default_config)
    mse = metrics.aggregation_mse(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
    )
    assert mse > 0


def test_metrics_verify_constraints(default_config):
    metrics = VFEELMetrics(default_config)
    result = metrics.verify_constraints(
        batch_size=150,
        sensing_power=0.5,
        transmit_powers=np.ones(default_config.K) * 0.3,
        E_budget=1.0,
        T_budget=100e-3,
    )
    assert result['energy_satisfied']
    assert result['latency_satisfied']


def test_metrics_latency_accuracy_tradeoff(default_config):
    metrics = VFEELMetrics(default_config)
    result = metrics.latency_accuracy_tradeoff(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
    )
    assert 'latency_ms' in result
    assert 'accuracy_proxy' in result
    assert 0 <= result['accuracy_proxy'] <= 1


# ─── ConvergenceAnalyzer ─────────────────────────────────────────────────────

def test_convergence_analyzer_rounds(default_config):
    analyzer = ConvergenceAnalyzer(default_config)
    rounds = analyzer.compute_round_complexity(
        batch_size=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
        target_accuracy=0.85,
    )
    assert isinstance(rounds, int)
    assert rounds > 0


def test_convergence_analyzer_speedup(default_config):
    analyzer = ConvergenceAnalyzer(default_config)
    speedup = analyzer.compute_speedup(
        batch_size_1=50,
        batch_size_2=150,
        sensing_power=1.0,
        transmit_powers=np.ones(default_config.K) * 0.5,
        denoising_factor=0.5,
        target_accuracy=0.85,
    )
    assert speedup > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
