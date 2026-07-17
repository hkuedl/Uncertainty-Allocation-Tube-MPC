"""Stochastic soft tube MPC solver."""

from __future__ import annotations

import numpy as np

from ..config import PhysicalParams
from .tube_mpc import TubeMPC


class StochasticTMPC(TubeMPC):
    """Soft-only stochastic TMPC with distributionally robust tube bounds."""

    bounds_non_dr = False

    def __init__(
        self,
        config,
        scenario_generator,
        physical_params: PhysicalParams | None = None,
        alphas: tuple[
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
        ] = (0.25, 0.25, 0.25, 0.25),
    ):
        super().__init__(config, scenario_generator, physical_params, alphas=alphas)

    def solve_soft_tube(
        self,
        x_data,
        price_data,
        t0: int,
        ess0: float,
        tau0: float,
        ww: float,
        method: str = "Uniform",
        beta: float | None = None,
    ):
        """Solve stochastic TMPC with slack variables."""
        return self._solve_tube(
            x_data=x_data,
            price_data=price_data,
            t0=t0,
            ess0=ess0,
            tau0=tau0,
            method=method,
            beta=beta,
            soft=True,
            ww=ww,
        )

    def solve_hard_tube(self, *args, **kwargs):
        """Hard-tube mode is intentionally unsupported for stochastic TMPC."""
        raise NotImplementedError("StochasticTMPC only supports soft tube constraints.")
