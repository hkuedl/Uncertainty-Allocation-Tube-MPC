"""Configuration objects shared across UATMPC modules."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Paths:
    """Filesystem locations for data, models, and experiment outputs."""

    root: Path
    solar_data: Path
    load_data: Path
    price_data: Path
    temp_model_dir: Path
    pv_model_dir: Path
    load_model_dir: Path
    results_dir: Path

    @classmethod
    def from_root(cls, root: str | Path = ".") -> "Paths":
        root = Path(root)
        return cls(
            root=root,
            solar_data=root / "23solar_data15.csv",
            load_data=root / "building data" / "2zonesupermarket15.csv",
            price_data=root / "United Kingdom.csv",
            temp_model_dir=root / "15temp_models",
            pv_model_dir=root / "15pv_models",
            load_model_dir=root / "15load_models",
            results_dir=root / "UATMPC" / "results",
        )


@dataclass(frozen=True)
class MPCConfig:
    """Numerical settings used by forecasting, scenarios, and MPC solvers."""

    control_horizon: int = 8
    context_steps: int = 24 * 4
    quantiles: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    beta: float = 0.95
    random_seed: int = 42
    n_scenarios: int = 50


@dataclass(frozen=True)
class PhysicalParams:
    """Physical and economic constants from final_single.ipynb."""

    c_ther: float = 0.5
    gama1: float = 0.2
    gama2: float = 0.3
    eta_heat: float = 0.95
    eta_cool: float = 0.8
    tau_ref: float = 26
    p_hvac_max: float = 20
    c_deg: float = 0.005
    eta_ch: float = 0.95
    eta_dch: float = 0.92
    ess_max: float = 20
    ess_min: float = 0
    p_ch_max: float = 30
    p_dis_max: float = 30
    p_ch_min: float = 0
    p_dis_min: float = 0
    www: float = 1

    @cached_property
    def tau_min(self) -> np.ndarray:
        return np.concatenate([22 * np.ones(8 * 4), 24 * np.ones(14 * 4), 22 * np.ones(6 * 4)])

    @cached_property
    def tau_max(self) -> np.ndarray:
        return np.concatenate([30 * np.ones(8 * 4), 28 * np.ones(14 * 4), 30 * np.ones(6 * 4)])

    @cached_property
    def tau_ini(self) -> float:
        return self.tau_ref

    @cached_property
    def a_temp(self) -> float:
        return 1 - self.gama1

    @cached_property
    def b_temp(self) -> np.ndarray:
        return np.array([self.gama2 * self.eta_heat, -self.gama2 * self.eta_cool]).reshape(1, -1)

    def thermal_power_effect(self, p_heat, p_cool):
        return self.gama2 * (self.eta_heat * p_heat - self.eta_cool * p_cool)

    @cached_property
    def k_temp(self) -> np.ndarray:
        return np.array([1, 2]).reshape(-1, 1)

    @cached_property
    def phi_temp(self) -> np.ndarray:
        return self.a_temp + self.b_temp @ self.k_temp

    @cached_property
    def a_ess(self) -> float:
        return 1

    @cached_property
    def b_ess(self) -> np.ndarray:
        return np.array([self.eta_ch, -1 / self.eta_dch]).reshape(1, -1)

    @cached_property
    def k_ess(self) -> np.ndarray:
        return np.array([0.05 / self.eta_ch, self.eta_dch]).reshape(-1, 1)

    @cached_property
    def phi_ess(self) -> np.ndarray:
        return self.a_ess + self.b_ess @ self.k_ess

    @cached_property
    def ess_int(self) -> float:
        return 0.5 * self.ess_max
