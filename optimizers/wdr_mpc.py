"""Weighted distributionally robust MPC solvers."""

from __future__ import annotations

import numpy as np

from ..ambiguity import dr_bounds
from ..config import PhysicalParams


class WDRMPC:
    """Hard- and soft-tube WDR-MPC entry points with fixed ESS disturbance response."""

    def __init__(self, config, scenario_generator, physical_params: PhysicalParams | None = None):
        self.config = config
        self.scenario_generator = scenario_generator
        self.params = physical_params or PhysicalParams()

    def solve_hard_tube(
        self,
        x_data,
        price_data,
        t0: int,
        ess0: float,
        tau0: float,
        method: str = "Uniform",
        beta: float | None = None,
    ):
        """Solve WDR-MPC with hard tubes."""
        return self._solve_tube(
            x_data=x_data,
            price_data=price_data,
            t0=t0,
            ess0=ess0,
            tau0=tau0,
            method=method,
            beta=beta,
            soft=False,
            ww=0.0,
        )

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
        """Solve WDR-MPC with slack variables."""
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

    def _solve_tube(
        self,
        x_data,
        price_data,
        t0: int,
        ess0: float,
        tau0: float,
        method: str,
        beta: float | None,
        soft: bool,
        ww: float,
    ):
        import cvxpy as cp

        c_buy, c_cur, _ = price_data
        params = self.params
        beta = self.config.beta if beta is None else beta

        tau_forecast, p_forecast, tau_scens, p_scens, weights = self.scenario_generator.generate(
            x_data,
            t0,
            method,
        )
        weights = np.asarray(weights)
        _, horizon = tau_scens.shape

        phi_ess = float(np.asarray(params.phi_ess).item())
        k_ess_ch = float(params.k_ess[0, 0])
        k_ess_dis = float(params.k_ess[1, 0])
        phi_temp = float(np.asarray(params.phi_temp).item())     
        k_temp_heat = float(params.k_temp[0, 0])
        k_temp_cool = float(params.k_temp[1, 0])

        e_ess = np.zeros(p_scens.shape)
        for i in range(p_scens.shape[0]):
            for t in range(horizon):
                if t == 0:
                    e_ess[i, t] = p_scens[i, t]
                else:
                    e_ess[i, t] = phi_ess * e_ess[i, t-1] + p_scens[i, t]
        
        tau_ess = np.zeros(tau_scens.shape)
        for i in range(tau_scens.shape[0]):
            for t in range(horizon):
                if t == 0:
                    tau_ess[i, t] = tau_scens[i, t]
                else:
                    tau_ess[i, t] = phi_temp * tau_ess[i, t-1] + tau_scens[i, t]

        
        s_ess_bounds = [dr_bounds(e_ess[:, h], weights, beta=beta) for h in range(horizon)] 
        s_temp_bounds = [dr_bounds(tau_ess[:, h], weights, beta=beta) for h in range(horizon)]
        
        s_ess_lb = np.zeros(horizon)
        s_ess_ub = np.zeros(horizon)
        for t in range(horizon):
            s_ess_lb[t] = s_ess_bounds[t][0]
            s_ess_ub[t] = s_ess_bounds[t][1]
            
        s_temp_lb = np.zeros(horizon)
        s_temp_ub = np.zeros(horizon)
        for t in range(horizon):
            s_temp_lb[t] = s_temp_bounds[t][0]
            s_temp_ub[t] = s_temp_bounds[t][1]
        
        p_buy_hat = cp.Variable(horizon, nonneg=True)
        p_ess_hat = cp.Variable(horizon)
        p_hvac_hat = cp.Variable(horizon, nonneg=True)
        p_pvc_hat = cp.Variable(horizon, nonneg=True)

        tau = cp.Variable(horizon)
        p_heat = cp.Variable(horizon, nonneg=True)
        p_cool = cp.Variable(horizon, nonneg=True)
        hvac_heat_mode = cp.Variable(horizon, boolean=True)

        ess = cp.Variable(horizon)
        p_ch = cp.Variable(horizon, nonneg=True)
        p_dis = cp.Variable(horizon, nonneg=True)
        ess_charge_mode = cp.Variable(horizon, boolean=True)

        constraints = []
        c_deg_cost = params.c_deg * (cp.sum(p_ch) * params.eta_ch + cp.sum(p_dis) / params.eta_dch)
        c_ther_cost = params.c_ther * cp.norm(tau - params.tau_ref, 2) ** 2
        c_buy_cost = c_buy[t0 : t0 + horizon] @ p_buy_hat
        c_cur_cost = c_cur * cp.sum(p_pvc_hat)

        if soft:
            tau_up_slack = cp.Variable(horizon, nonneg=True)
            tau_lo_slack = cp.Variable(horizon, nonneg=True)
            ess_up_slack = cp.Variable(horizon, nonneg=True)
            ess_lo_slack = cp.Variable(horizon, nonneg=True)
            c_slack = cp.sum(ess_up_slack + ess_lo_slack + tau_up_slack + tau_lo_slack)
            objective = cp.Minimize(c_deg_cost + c_ther_cost + c_buy_cost + c_cur_cost + ww * c_slack)
        else:
            tau_up_slack = tau_lo_slack = ess_up_slack = ess_lo_slack = None
            objective = cp.Minimize(c_deg_cost + c_ther_cost + c_buy_cost + c_cur_cost)

        constraints.append(p_forecast == p_buy_hat - p_ess_hat - p_hvac_hat - p_pvc_hat)

        constraints.append(p_ess_hat == p_ch - p_dis)
        constraints.append(p_ch <= params.p_ch_max * ess_charge_mode)
        constraints.append(p_dis <= params.p_dis_max * (1 - ess_charge_mode))

        for t in range(horizon):
            if soft:
                constraints.append(ess[t] <= params.ess_max)
                constraints.append(ess[t] >= params.ess_min)
                constraints.append(ess[t] - ess_up_slack[t] <= params.ess_max - s_ess_ub[t])
                constraints.append(ess[t] + ess_lo_slack[t] >= params.ess_min - s_ess_lb[t])
            else:
                constraints.append(ess[t] <= params.ess_max - s_ess_ub[t])
                constraints.append(ess[t] >= params.ess_min - s_ess_lb[t])
            constraints.append(p_ch[t] <= params.p_ch_max - k_ess_ch * s_ess_ub[t] * params.eta_ch)
            constraints.append(p_dis[t] <= params.p_dis_max + k_ess_dis * s_ess_lb[t] / params.eta_dch)

            if t == 0:
                constraints.append(ess[t] == ess0 + params.eta_ch * p_ch[t] - p_dis[t] / params.eta_dch)
            else:
                constraints.append(
                    ess[t] == ess[t - 1] + params.eta_ch * p_ch[t] - p_dis[t] / params.eta_dch
                )
            if t + t0 == 24 * 4 - 1:
                constraints.append(ess[t] == params.ess_int)

        constraints.append(p_hvac_hat == p_heat + p_cool)
        constraints.append(p_heat <= params.p_hvac_max * hvac_heat_mode)
        constraints.append(p_cool <= params.p_hvac_max * (1 - hvac_heat_mode))

        for t in range(horizon):
            if soft:
                constraints.append(tau[t] <= params.tau_max[t0 + t])
                constraints.append(tau[t] >= params.tau_min[t0 + t])
                constraints.append(tau[t] - tau_up_slack[t] <= params.tau_max[t0 + t] - s_temp_ub[t])
                constraints.append(tau[t] + tau_lo_slack[t] >= params.tau_min[t0 + t] - s_temp_lb[t])
            else:
                constraints.append(tau[t] <= params.tau_max[t0 + t] - s_temp_ub[t])
                constraints.append(tau[t] >= params.tau_min[t0 + t] - s_temp_lb[t])
            constraints.append(p_heat[t] <= params.p_hvac_max - k_temp_heat * s_temp_ub[t])
            constraints.append(p_cool[t] <= params.p_hvac_max + k_temp_cool * s_temp_lb[t])

            if t == 0:
                constraints.append(
                    tau[t]
                    == (1 - params.gama1) * tau0
                    + params.thermal_power_effect(p_heat[t], p_cool[t])
                    + params.gama1 * tau_forecast[t]
                )
            else:
                constraints.append(
                    tau[t]
                    == (1 - params.gama1) * tau[t - 1]
                    + params.thermal_power_effect(p_heat[t], p_cool[t])
                    + params.gama1 * tau_forecast[t]
                )
            if t + t0 == 24 * 4 - 1:
                constraints.append(tau[t] == params.tau_ini)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)
        # print(f"Current Iteration: {t0} Problem status: {problem.status}")
        if soft and problem.status != "optimal":
            print(s_ess_bounds)
            print(s_temp_bounds)

        forecasts = [p_forecast[0], tau_forecast[0]]
        controls = [p_buy_hat.value[0], p_pvc_hat.value[0]]
        ther_ps = [
            p_hvac_hat.value[0],
            p_heat.value[0],
            p_cool.value[0],
            params.p_hvac_max - k_temp_heat * s_temp_ub[0],
            params.p_hvac_max + k_temp_cool * s_temp_lb[0],
        ]
        ther_taus = [
            tau.value[0],
            params.tau_max[t0] - s_temp_ub[0],
            params.tau_min[t0] - s_temp_lb[0],
        ]
        ess_ps = [
            p_ess_hat.value[0],
            p_ch.value[0],
            p_dis.value[0],
            params.p_ch_max - k_ess_ch * s_ess_ub[0],
            params.p_dis_max + k_ess_dis * s_ess_lb[0],
        ]
        ess_esss = [
            ess.value[0],
            params.ess_max - s_ess_ub[0],
            params.ess_min - s_ess_lb[0],
        ]
        return forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss
