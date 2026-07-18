"""Distributionally robust MPC solvers."""

from __future__ import annotations

import numpy as np


from ..ambiguity import dr_bounds, radius
from ..config import PhysicalParams


class DRMPC:
    """Distributionally robust MPC solver."""
    
    def __init__(self, config, scenario_generator, physical_params: PhysicalParams | None = None):
        self.config = config
        self.scenario_generator = scenario_generator
        self.params = physical_params or PhysicalParams()

    def solve(self, x_data, price_data, t0: int, ess0: float, tau0: float, method: str = "Uniform", beta: float | None = None):
        """Solve the distributionally robust MPC problem."""
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
        n_samples, horizon = tau_scens.shape
        # The ambiguity set radius is calculated on ell_1 norm, which is the coordinate-wise sum.
        # elec_epi_total = np.sum(np.array([radius(p_scens[:, h], beta=beta) for h in range(horizon)]))
        # temp_epi_total = np.sum(np.array([radius(tau_scens[:, h], beta=beta) for h in range(horizon)]))
        # print(elec_epi_total, temp_epi_total)
        elec_epi_total = 1
        temp_epi_total = 0.1

        p_buy_hat = cp.Variable(horizon, nonneg=True)
        p_ess_hat = cp.Variable(horizon)
        p_hvac_hat = cp.Variable(horizon, nonneg=True)
        p_pvc_hat = cp.Variable(horizon, nonneg=True)

        nom_tau = cp.Variable(horizon)
        p_heat = cp.Variable(horizon, nonneg=True)
        p_cool = cp.Variable(horizon, nonneg=True)
        hvac_heat_mode = cp.Variable(horizon, boolean=True)

        nom_ess = cp.Variable(horizon)
        p_ch = cp.Variable(horizon, nonneg=True)
        p_dis = cp.Variable(horizon, nonneg=True)
        ess_charge_mode = cp.Variable(horizon, boolean=True)
        
        c_deg_cost = params.c_deg * (cp.sum(p_ch) * params.eta_ch + cp.sum(p_dis) / params.eta_dch)
        c_ther_cost = params.c_ther * cp.norm(nom_tau - params.tau_ref, 2) ** 2
        c_buy_cost = c_buy[t0 : t0 + horizon] @ p_buy_hat
        c_cur_cost = c_cur * cp.sum(p_pvc_hat)

        # slack variables for DRCC
        ess_slack = cp.Variable(nonneg=True)
        thermal_slack = cp.Variable(nonneg=True)
        objective = cp.Minimize(c_buy_cost + c_deg_cost + c_cur_cost + c_ther_cost + (1.0*ess_slack + 1.0*thermal_slack))

        constraints = []
        # nominal ESS dynamics
        for t in range(horizon):
            if t == 0:
                constraints.append(nom_ess[t] == ess0 + params.eta_ch * p_ch[t] - p_dis[t] / params.eta_dch)
            else:
                constraints.append(nom_ess[t] == nom_ess[t - 1] + params.eta_ch * p_ch[t] - p_dis[t] / params.eta_dch)
            if t + t0 == 24 * 4 - 1:
                constraints.append(nom_ess[t] == params.ess_int)

        constraints.append(nom_ess <= params.ess_max)
        constraints.append(nom_ess >= params.ess_min)
        constraints.append(p_ch <= params.p_ch_max)
        constraints.append(p_dis <= params.p_dis_max)
        constraints.append(p_ch <= params.p_ch_max * ess_charge_mode)
        constraints.append(p_dis <= params.p_dis_max * (1 - ess_charge_mode))
        constraints.append(p_ess_hat == p_ch - p_dis)

        # nominal thermal dynamics  
        for t in range(horizon):
            if t == 0:
                constraints.append(nom_tau[t] == tau0 + params.gama1 * (tau_forecast[t] - tau0) + params.thermal_power_effect(p_heat[t], p_cool[t]))
            else:
                constraints.append(nom_tau[t] == nom_tau[t - 1] + params.gama1 * (tau_forecast[t] - tau_forecast[t - 1]) + params.thermal_power_effect(p_heat[t], p_cool[t]))
            if t + t0 == 24 * 4 - 1:
                constraints.append(nom_tau[t] == params.tau_ini)
            constraints.append(nom_tau[t] <= params.tau_max[t0+t])
            constraints.append(nom_tau[t] >= params.tau_min[t0+t])
        constraints.append(p_heat <= params.p_hvac_max)
        constraints.append(p_cool <= params.p_hvac_max)
        constraints.append(p_heat <= params.p_hvac_max * hvac_heat_mode)
        constraints.append(p_cool <= params.p_hvac_max * (1 - hvac_heat_mode))
        constraints.append(p_hvac_hat == p_heat + p_cool)
        constraints.append(p_forecast + p_ess_hat + p_hvac_hat + p_pvc_hat == p_buy_hat)

        # distributionally robust chance constraints
        # ESS uncertainty propagation matrix
        R_ess = np.tril(np.ones((horizon, horizon)))
        A_ess = np.vstack([R_ess, -R_ess])
        eta_ess = cp.Variable()
        b_ess = cp.hstack([nom_ess - params.ess_max - eta_ess, -nom_ess + params.ess_min - eta_ess])
        lam_ess = cp.Variable(nonneg=True)
        s_ess = cp.Variable(n_samples, nonneg=True)
        constraints.append(eta_ess + (lam_ess * elec_epi_total + cp.sum(s_ess)/n_samples)/(1-beta) <= ess_slack)
        for k in range(A_ess.shape[0]):
            for i in range(n_samples):
                constraints.append(s_ess[i] >= A_ess[k, :] @ p_scens[i, :] + b_ess[k])
            constraints.append(lam_ess >= np.linalg.norm(A_ess[k, :], ord=np.inf))

        # thermal uncertainty matrix
        R_ther = np.tril(np.ones((horizon, horizon))) * params.gama1
        for m in range(horizon):
            for n in range(m):
                R_ther[m, n] = (1-params.gama1)*R_ther[m-1, n]
        A_ther = np.vstack([R_ther, -R_ther])
        eta_ther = cp.Variable()
        b_ther = cp.hstack([nom_tau - params.tau_max[t0:t0+horizon] - eta_ther, -nom_tau + params.tau_min[t0:t0+horizon] - eta_ther])
        lam_ther = cp.Variable(nonneg=True)
        s_ther = cp.Variable(n_samples, nonneg=True)
        constraints.append(eta_ther + (lam_ther * temp_epi_total + cp.sum(s_ther)/n_samples)/(1-beta) <= thermal_slack)
        for k in range(A_ther.shape[0]):
            for i in range(n_samples):
                constraints.append(s_ther[i] >= A_ther[k, :] @ tau_scens[i, :] + b_ther[k])
            constraints.append(lam_ther >= np.linalg.norm(A_ther[k, :], ord=np.inf))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.GUROBI)
        # print(f"Current time: {t0}. Problem Status: {problem.status}")

        forecasts = [p_forecast[0], tau_forecast[0]]
        controls = [p_buy_hat.value[0], p_pvc_hat.value[0]]
        ther_ps = [p_hvac_hat.value[0], p_heat.value[0], p_cool.value[0]]
        ther_taus = [nom_tau.value[0]]
        ess_ps = [p_ess_hat.value[0], p_ch.value[0], p_dis.value[0]]
        ess_esss = [nom_ess.value[0]]
        return forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss