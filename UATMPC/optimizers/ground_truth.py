"""Full-day MPC with ground-truth future information."""

from __future__ import annotations

import numpy as np

from ..config import PhysicalParams


class GroundTruthMPC:
    """Ground-truth full-day benchmark using a fixed 24-hour horizon."""

    def __init__(
        self,
        config,
        physical_params: PhysicalParams | None = None,
        horizon: int = 24 * 4,
    ):
        self.config = config
        self.params = physical_params or PhysicalParams()
        self.horizon = horizon

    def solve(
        self,
        real_data,
        price_data,
        t0: int = 0,
        ess0: float | None = None,
        tau0: float | None = None,
        method=None,
    ):
        """Solve the 96-step ground-truth optimization problem."""
        import cvxpy as cp

        real_temp, real_net_load = real_data
        c_buy, c_cur, _ = price_data
        params = self.params
        horizon = self.horizon
        ess0 = params.ess_int if ess0 is None else ess0
        tau0 = params.tau_ini if tau0 is None else tau0

        tau_forecast = np.asarray(real_temp[t0 : t0 + horizon]).reshape(-1)
        p_forecast = np.asarray(real_net_load[t0 : t0 + horizon]).reshape(-1)
        c_buy_horizon = np.asarray(c_buy[t0 : t0 + horizon]).reshape(-1)
        tau_max = np.asarray(params.tau_max[t0 : t0 + horizon]).reshape(-1)
        tau_min = np.asarray(params.tau_min[t0 : t0 + horizon]).reshape(-1)

        available_steps = min(
            len(tau_forecast),
            len(p_forecast),
            len(c_buy_horizon),
            len(tau_max),
            len(tau_min),
        )
        if available_steps < horizon:
            raise ValueError("GroundTruthMPC requires 96 available steps from t0.")

        p_buy_hat = cp.Variable(horizon, nonneg=True)
        p_pvc_hat = cp.Variable(horizon, nonneg=True)
        p_ch_hat = cp.Variable(horizon, nonneg=True)
        p_dis_hat = cp.Variable(horizon, nonneg=True)
        p_ess_hat = cp.Variable(horizon)
        p_hvac_hat = cp.Variable(horizon)
        p_heat_hat = cp.Variable(horizon, nonneg=True)
        p_cool_hat = cp.Variable(horizon, nonneg=True)
        hvac_heat_mode = cp.Variable(horizon, boolean=True)
        tau_hat = cp.Variable(horizon)
        ess_hat = cp.Variable(horizon)
        ess_charge_mode = cp.Variable(horizon, boolean=True)

        c_buy_cost = c_buy_horizon @ p_buy_hat
        c_deg_cost = params.c_deg * (
            params.eta_ch * cp.sum(p_ch_hat) + cp.sum(p_dis_hat) / params.eta_dch
        )
        c_cur_cost = c_cur * cp.sum(p_pvc_hat)
        c_ther_cost = params.c_ther * cp.norm(tau_hat - params.tau_ref, 2) ** 2
        objective = cp.Minimize(c_buy_cost + c_deg_cost + c_cur_cost + c_ther_cost)

        constraints = []
        for t in range(horizon):
            if t == 0:
                constraints.append(
                    ess_hat[t] == ess0 + params.eta_ch * p_ch_hat[t] - p_dis_hat[t] / params.eta_dch
                )
            else:
                constraints.append(
                    ess_hat[t]
                    == ess_hat[t - 1] + params.eta_ch * p_ch_hat[t] - p_dis_hat[t] / params.eta_dch
                )
            if t + t0 == 4 * 24 - 1:
                constraints.append(ess_hat[t] == params.ess_int)

        constraints.append(ess_hat <= params.ess_max)
        constraints.append(ess_hat >= params.ess_min)
        constraints.append(p_ch_hat <= params.p_ch_max)
        constraints.append(p_dis_hat <= params.p_dis_max)
        constraints.append(p_ch_hat <= params.p_ch_max * ess_charge_mode)
        constraints.append(p_dis_hat <= params.p_dis_max * (1 - ess_charge_mode))
        constraints.append(p_ess_hat == p_ch_hat - p_dis_hat)

        for t in range(horizon):
            if t == 0:
                constraints.append(
                    tau_hat[t]
                    == tau0
                    + params.gama1 * (tau_forecast[t] - tau0)
                    + params.thermal_power_effect(p_heat_hat[t], p_cool_hat[t])
                )
            else:
                constraints.append(
                    tau_hat[t]
                    == tau_hat[t - 1]
                    + params.gama1 * (tau_forecast[t] - tau_hat[t - 1])
                    + params.thermal_power_effect(p_heat_hat[t], p_cool_hat[t])
                )
            if t + t0 == 4 * 24 - 1:
                constraints.append(tau_hat[t] == params.tau_ini)
            constraints.append(tau_hat[t] <= tau_max[t])
            constraints.append(tau_hat[t] >= tau_min[t])

        constraints.append(p_heat_hat <= params.p_hvac_max)
        constraints.append(p_cool_hat <= params.p_hvac_max)
        constraints.append(p_heat_hat <= params.p_hvac_max * hvac_heat_mode)
        constraints.append(p_cool_hat <= params.p_hvac_max * (1 - hvac_heat_mode))
        constraints.append(p_hvac_hat == p_heat_hat + p_cool_hat)
        constraints.append(p_forecast + p_ess_hat + p_hvac_hat + p_pvc_hat == p_buy_hat)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"GroundTruthMPC solve failed with status: {problem.status}")

        forecasts = [p_forecast, tau_forecast]
        controls = [p_buy_hat.value, p_pvc_hat.value]
        ther_ps = [p_hvac_hat.value, p_heat_hat.value, p_cool_hat.value]
        ther_taus = [tau_hat.value]
        ess_ps = [p_ess_hat.value, p_ch_hat.value, p_dis_hat.value]
        ess_esss = [ess_hat.value]
        return forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss
