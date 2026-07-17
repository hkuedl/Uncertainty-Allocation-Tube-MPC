"""Tube-based MPC solvers."""

from __future__ import annotations

import numpy as np

from ..ambiguity import dr_bounds, radius
from ..config import PhysicalParams


class TubeMPC:
    """Tube-based MPC solver with fixed affine feedback allocation."""

    bounds_non_dr = True

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
        self.config = config
        self.scenario_generator = scenario_generator
        self.params = physical_params or PhysicalParams()
        self.alphas = alphas


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
        """Solve tube MPC with hard tubes."""
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
        """Solve tube MPC with slack variables."""
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

    def _alpha_vectors(self, horizon: int):
        alphas = [np.asarray(alpha, dtype=float) for alpha in self.alphas]
        alpha_vectors = []
        for alpha in alphas:
            if alpha.ndim == 0:
                alpha_vectors.append(np.full(horizon, float(alpha)))
            elif alpha.shape == (horizon,):
                alpha_vectors.append(alpha)
            else:
                raise ValueError("Each alpha must be a scalar or an array with length equal to the horizon.")

        alpha_sum = sum(alpha_vectors)
        if not np.allclose(alpha_sum, np.ones(horizon)):
            raise ValueError("Fixed alphas must sum to 1 at every horizon step.")
        if any(np.any(alpha < 0) for alpha in alpha_vectors):
            raise ValueError("Fixed alphas must be nonnegative.")
        return alpha_vectors

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
        n_samples, horizon = tau_scens.shape

        alph_buy, alph_hvac, alph_ess, alph_pvc = self._alpha_vectors(horizon)

        elec_w = [
            dr_bounds(params.www * p_scens[:, h], weights, beta=beta, non_dr=self.bounds_non_dr)
            for h in range(horizon)
        ]
        temp_w = [
            dr_bounds(params.www * tau_scens[:, h], weights, beta=beta, non_dr=self.bounds_non_dr)
            for h in range(horizon)
        ]
        elec_epi = np.array([radius(p_scens[:, h], beta=beta) for h in range(horizon)])

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

        cc = alph_buy * c_buy[t0 : t0 + horizon] - alph_pvc * c_cur
        c_exp = np.max(np.abs(cc)) * np.sum(elec_epi) + np.sum((p_scens * cc).T @ weights)

        if soft:
            tau_up_slack = cp.Variable(horizon, nonneg=True)
            tau_lo_slack = cp.Variable(horizon, nonneg=True)
            ess_up_slack = cp.Variable(horizon, nonneg=True)
            ess_lo_slack = cp.Variable(horizon, nonneg=True)
            c_slack = cp.sum(ess_up_slack + ess_lo_slack + tau_up_slack + tau_lo_slack)
            objective = cp.Minimize(c_deg_cost + c_ther_cost + c_buy_cost + c_cur_cost + c_exp + ww * c_slack)
        else:
            tau_up_slack = tau_lo_slack = ess_up_slack = ess_lo_slack = None
            objective = cp.Minimize(c_deg_cost + c_ther_cost + c_buy_cost + c_cur_cost + c_exp)

        a_ee_cc = cp.Variable((horizon, 2))
        b_ee_cc = cp.Variable((horizon, 2))
        eta_ee_cc = cp.Variable(horizon)
        s_ee_cc = cp.Variable((horizon, n_samples), nonneg=True)
        lam_ee_cc = cp.Variable(horizon)
        for t in range(horizon):
            constraints.append(a_ee_cc[t, 0] == -alph_buy[t])
            constraints.append(a_ee_cc[t, 1] == alph_pvc[t])
            constraints.append(b_ee_cc[t, 0] == -p_buy_hat[t] - eta_ee_cc[t])
            constraints.append(b_ee_cc[t, 1] == -p_pvc_hat[t] - eta_ee_cc[t])
            constraints.append(
                eta_ee_cc[t]
                + (lam_ee_cc[t] * elec_epi[t] + weights @ s_ee_cc[t, :]) / (1 - beta)
                <= 0
            )
            constraints.append(lam_ee_cc[t] >= cp.norm(a_ee_cc[t, :], np.inf))
            for i in range(n_samples):
                constraints.append(
                    s_ee_cc[t, i] >= a_ee_cc[t, 0] * p_scens[i, t] + b_ee_cc[t, 0]
                )
                constraints.append(
                    s_ee_cc[t, i] >= a_ee_cc[t, 1] * p_scens[i, t] + b_ee_cc[t, 1]
                )

        constraints.append(p_forecast == p_buy_hat - p_ess_hat - p_hvac_hat - p_pvc_hat)

        constraints.append(p_ess_hat == p_ch - p_dis)
        constraints.append(p_ch <= params.p_ch_max * ess_charge_mode)
        constraints.append(p_dis <= params.p_dis_max * (1 - ess_charge_mode))
        phi_ess = float(np.asarray(params.phi_ess).item())
        k_ess_ch = float(params.k_ess[0, 0])
        k_ess_dis = float(params.k_ess[1, 0])
        s_ess_lb = np.zeros(horizon)
        s_ess_ub = np.zeros(horizon)
        for t, bounds in enumerate(elec_w):
            e_lb, e_ub = bounds 
            soc_error_ub = e_ub / params.eta_dch if e_ub > 0 else e_ub * params.eta_ch
            soc_error_lb = e_lb / params.eta_dch if e_lb > 0 else e_lb * params.eta_ch
            if t == 0:
                s_ess_lb[t] = alph_ess[t] * soc_error_lb
                s_ess_ub[t] = alph_ess[t] * soc_error_ub
            else:
                s_ess_lb[t] = phi_ess * s_ess_lb[t - 1] + alph_ess[t] * soc_error_lb
                s_ess_ub[t] = phi_ess * s_ess_ub[t - 1] + alph_ess[t] * soc_error_ub
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
        phi_temp = float(np.asarray(params.phi_temp).item())
        k_temp_heat = float(params.k_temp[0, 0])
        k_temp_cool = float(params.k_temp[1, 0])
        s_temp_lb = np.zeros(horizon)
        s_temp_ub = np.zeros(horizon)
        for t, bounds in enumerate(temp_w):
            elec_bounds = elec_w[t]
            if t == 0:
                s_temp_lb[t] = (
                    params.gama1 * bounds[0]
                    + params.gama2 * params.eta_cool * elec_bounds[0] * alph_hvac[t]
                )
                s_temp_ub[t] = (
                    params.gama1 * bounds[1]
                    + params.gama2 * params.eta_heat * elec_bounds[1] * alph_hvac[t]
                )
            else:
                s_temp_lb[t] = (
                    phi_temp * s_temp_lb[t - 1]
                    + params.gama1 * bounds[0]
                    + params.gama2 * params.eta_cool * elec_bounds[0] * alph_hvac[t]
                )
                s_temp_ub[t] = (
                    phi_temp * s_temp_ub[t - 1]
                    + params.gama1 * bounds[1]
                    + params.gama2 * params.eta_heat * elec_bounds[1] * alph_hvac[t]
                )

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
            print(elec_w)
            print(temp_w)

        forecasts = [p_forecast[0], tau_forecast[0]]
        controls = [p_buy_hat.value[0], p_pvc_hat.value[0]]
        fixed_alphas = [alph_buy[0], alph_hvac[0], alph_ess[0], alph_pvc[0]]
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
        return forecasts, controls, fixed_alphas, ther_ps, ther_taus, ess_ps, ess_esss