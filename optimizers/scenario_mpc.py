"""Scenario-based MPC benchmark."""

from __future__ import annotations

import numpy as np

from ..config import PhysicalParams


class ScenarioMPC:
    """Scenario-based MPC solver."""

    def __init__(self, config, scenario_generator, physical_params: PhysicalParams | None = None):
        self.config = config
        self.scenario_generator = scenario_generator
        self.params = physical_params or PhysicalParams()

    def solve(self, x_data, price_data, t0: int, ess0: float, tau0: float, method: str = "Uniform"):
        """Solve the scenario-based MPC benchmark."""
        import cvxpy as cp

        c_buy, c_cur, _ = price_data
        params = self.params

        tau_forecast, p_forecast, tau_scens, p_scens, weights = self.scenario_generator.generate(
            x_data,
            t0,
            method,
        )
        weights = np.asarray(weights)
        tau_scens = tau_scens + tau_forecast
        p_scens = params.www * p_scens + p_forecast
        n_samples, horizon = tau_scens.shape

        p_buy = cp.Variable((n_samples, horizon), nonneg=True)
        p_pvc = cp.Variable((n_samples, horizon), nonneg=True)
        p_ch = cp.Variable((n_samples, horizon), nonneg=True)
        p_dis = cp.Variable((n_samples, horizon), nonneg=True)
        ess_charge_mode = cp.Variable((n_samples, horizon), boolean=True)
        p_ess = cp.Variable((n_samples, horizon))
        ess = cp.Variable((n_samples, horizon))
        p_heat = cp.Variable((n_samples, horizon), nonneg=True)
        p_cool = cp.Variable((n_samples, horizon), nonneg=True)
        hvac_heat_mode = cp.Variable((n_samples, horizon), boolean=True)
        p_hvac = cp.Variable((n_samples, horizon))
        tau = cp.Variable((n_samples, horizon))

        p_buy_hat = cp.Variable(nonneg=True)
        p_pvc_hat = cp.Variable(nonneg=True)
        p_ch_hat = cp.Variable(nonneg=True)
        p_dis_hat = cp.Variable(nonneg=True)
        p_ess_hat = cp.Variable()
        p_heat_hat = cp.Variable(nonneg=True)
        p_cool_hat = cp.Variable(nonneg=True)
        p_hvac_hat = cp.Variable()

        c_buy_cost = cp.Variable(n_samples, nonneg=True)
        c_deg_cost = cp.Variable(n_samples, nonneg=True)
        c_cur_cost = cp.Variable(n_samples, nonneg=True)
        ther_diff = cp.Variable(n_samples, nonneg=True)
        power_diff = cp.Variable(n_samples)

        constraints = []
        for i in range(n_samples):
            constraints.append(c_buy_cost[i] == c_buy[t0 : t0 + horizon] @ p_buy[i, :])
            constraints.append(
                c_deg_cost[i]
                == params.c_deg
                * cp.sum(p_ch[i, :] * params.eta_ch + p_dis[i, :] / params.eta_dch)
            )
            constraints.append(c_cur_cost[i] == c_cur * cp.sum(p_pvc[i, :]))

        square_tau_diff = cp.square(tau - params.tau_ref)
        square_power_diff = cp.square(power_diff)
        ther_loss = cp.square(ther_diff)
        objective = cp.Minimize(
            weights
            @ (
                c_buy_cost
                + c_deg_cost
                + c_cur_cost
                + params.c_ther * cp.sum(square_tau_diff, axis=1)
            )
            + 100 * cp.sum(square_power_diff)
            + 100 * cp.sum(ther_loss)
        )

        for i in range(n_samples):
            constraints.append(p_buy_hat == p_buy[i, 0])
            constraints.append(p_pvc_hat == p_pvc[i, 0])
            constraints.append(p_ch_hat == p_ch[i, 0])
            constraints.append(p_dis_hat == p_dis[i, 0])
            constraints.append(p_ess_hat == p_ess[i, 0])
            constraints.append(p_heat_hat == p_heat[i, 0])
            constraints.append(p_cool_hat == p_cool[i, 0])
            constraints.append(p_hvac_hat == p_hvac[i, 0])

            constraints.append(
                p_buy[i, 1:]
                == p_scens[i, 1:] + p_ess[i, 1:] + p_pvc[i, 1:] + p_hvac[i, 1:]
            )
            constraints.append(
                power_diff[i]
                == p_buy[i, 0] - p_scens[i, 0] - p_ess[i, 0] - p_pvc[i, 0] - p_hvac[i, 0]
            )

            constraints.append(p_ess[i, :] == p_ch[i, :] - p_dis[i, :])
            constraints.append(p_ch[i, :] <= params.p_ch_max)
            constraints.append(p_dis[i, :] <= params.p_ch_max)
            constraints.append(p_ch[i, :] <= params.p_ch_max * ess_charge_mode[i, :])
            constraints.append(p_dis[i, :] <= params.p_dis_max * (1 - ess_charge_mode[i, :]))
            constraints.append(ess[i, :] <= params.ess_max)
            constraints.append(ess[i, :] >= params.ess_min)

            for t in range(horizon):
                if t == 0:
                    constraints.append(
                        ess[i, t]
                        == ess0 + params.eta_ch * p_ch[i, t] - p_dis[i, t] / params.eta_dch
                    )
                else:
                    constraints.append(
                        ess[i, t]
                        == ess[i, t - 1]
                        + params.eta_ch * p_ch[i, t]
                        - p_dis[i, t] / params.eta_dch
                    )
                if t + t0 == 24 * 4 - 1:
                    constraints.append(ess[i, t] == params.ess_int)
                constraints.append(tau[i, t] <= params.tau_max[t0 + t])
                constraints.append(tau[i, t] >= params.tau_min[t0 + t])

            constraints.append(p_hvac[i, :] == p_heat[i, :] + p_cool[i, :])
            constraints.append(p_heat[i, :] <= params.p_hvac_max)
            constraints.append(p_cool[i, :] <= params.p_hvac_max)
            constraints.append(p_heat[i, :] <= params.p_hvac_max * hvac_heat_mode[i, :])
            constraints.append(p_cool[i, :] <= params.p_hvac_max * (1 - hvac_heat_mode[i, :]))

            for t in range(horizon):
                if t == 0:
                    constraints.append(
                        tau[i, t]
                        == tau0
                        + params.gama1 * (tau_scens[i, t] - tau0)
                        + params.thermal_power_effect(p_heat[i, t], p_cool[i, t])
                    )
                else:
                    constraints.append(
                        tau[i, t]
                        == tau[i, t - 1]
                        + params.gama1 * (tau_scens[i, t] - tau[i, t - 1])
                        + params.thermal_power_effect(p_heat[i, t], p_cool[i, t])
                    )
                if t + t0 == 24 * 4 - 1:
                    constraints.append(ther_diff[i] == tau[i, t] - params.tau_ini)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)
        # print(f"Current time: {t0}, problem status: {problem.status}")

        tau_hat = weights @ tau.value
        ess_hat = weights @ ess.value
        forecasts = [p_forecast[0], tau_forecast[0]]
        controls = [p_buy_hat.value, p_pvc_hat.value]
        ther_ps = [p_hvac_hat.value, p_heat_hat.value, p_cool_hat.value]
        ther_taus = [tau_hat[0]]
        ess_ps = [p_ess_hat.value, p_ch_hat.value, p_dis_hat.value]
        ess_esss = [ess_hat[0]]
        return forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss
