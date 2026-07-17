"""Standard MPC with forecast predictions."""

from __future__ import annotations

from ..config import PhysicalParams


class StandardMPC:
    """Standard prediction-based MPC solver."""

    def __init__(self, config, scenario_generator, physical_params: PhysicalParams | None = None):
        self.config = config
        self.scenario_generator = scenario_generator
        self.params = physical_params or PhysicalParams()

    def solve(self, x_data, price_data, t0: int, ess0: float, tau0: float, method=None):
        """Solve the standard MPC problem."""
        import cvxpy as cp

        c_buy, c_cur, _ = price_data
        params = self.params

        tau_forecast, p_forecast, _, _, _ = self.scenario_generator.generate(x_data, t0)
        horizon = tau_forecast.shape[0]

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

        c_buy_cost = c_buy[t0 : t0 + self.config.control_horizon] @ p_buy_hat
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
            if t + t0 == 24 * 4 - 1:
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
            if t + t0 == 24 * 4 - 1:
                constraints.append(tau_hat[t] == params.tau_ini)

        constraints.append(tau_hat <= params.tau_max[t0 : t0 + horizon])
        constraints.append(tau_hat >= params.tau_min[t0 : t0 + horizon])
        constraints.append(p_heat_hat <= params.p_hvac_max)
        constraints.append(p_cool_hat <= params.p_hvac_max)
        constraints.append(p_heat_hat <= params.p_hvac_max * hvac_heat_mode)
        constraints.append(p_cool_hat <= params.p_hvac_max * (1 - hvac_heat_mode))
        constraints.append(p_hvac_hat == p_heat_hat + p_cool_hat)
        constraints.append(p_forecast + p_ess_hat + p_hvac_hat + p_pvc_hat == p_buy_hat)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)
        # print(f"Current time: {t0}. Problem Status: {problem.status}")

        forecasts = [p_forecast[0], tau_forecast[0]]
        controls = [p_buy_hat.value[0], p_pvc_hat.value[0]]
        ther_ps = [p_hvac_hat.value[0], p_heat_hat.value[0], p_cool_hat.value[0]]
        ther_taus = [tau_hat.value[0]]
        ess_ps = [p_ess_hat.value[0], p_ch_hat.value[0], p_dis_hat.value[0]]
        ess_esss = [ess_hat.value[0]]
        return forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss
