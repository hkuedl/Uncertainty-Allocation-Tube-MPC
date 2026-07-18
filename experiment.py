"""High-level experiment orchestration."""

from __future__ import annotations

import time

import numpy as np

from .config import MPCConfig, Paths, PhysicalParams
from .data import build_case_data, build_data_y_window, elec_price_read, load_all_data
from .forecasting import QuantileForecaster
from .optimizers import (
    DRMPC,
    GroundTruthMPC,
    PerfectPredictionMPC,
    RobustMPC,
    ScenarioMPC,
    StandardMPC,
    StochasticTMPC,
    TubeMPC,
    UATubeMPC,
    WDRMPC,
)
from .scenarios import ScenarioGenerator


class ExperimentRunner:
    """Load dependencies and run UATMPC comparison experiments."""

    def __init__(self, paths: Paths, config: MPCConfig, physical_params: PhysicalParams | None = None):
        self.paths = paths
        self.config = config
        self.physical_params = physical_params or PhysicalParams()
        self.data = None
        self.scenario_generator = None
        self.optimizers = {}

    def setup(self) -> "ExperimentRunner":
        """Load data, load forecasting models, and initialize solver objects."""
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.data = load_all_data(self.paths, self.config)

        temp_model = QuantileForecaster(self.paths.temp_model_dir).load()
        pv_model = QuantileForecaster(self.paths.pv_model_dir).load()
        load_model = QuantileForecaster(self.paths.load_model_dir).load()
        self.scenario_generator = ScenarioGenerator(temp_model, pv_model, load_model, self.config)

        self.optimizers = {
            "ua_tube": UATubeMPC(self.config, self.scenario_generator, self.physical_params),
            "tube": TubeMPC(self.config, self.scenario_generator, self.physical_params),
            "stochastic_tmpc": StochasticTMPC(self.config, self.scenario_generator, self.physical_params),
            "wdr": WDRMPC(self.config, self.scenario_generator, self.physical_params),
            "standard": StandardMPC(self.config, self.scenario_generator, self.physical_params),
            "perfect": PerfectPredictionMPC(self.config, self.physical_params),
            "ground_truth": GroundTruthMPC(self.config, self.physical_params),
            "scenario": ScenarioMPC(self.config, self.scenario_generator, self.physical_params),
            "dr": DRMPC(self.config, self.scenario_generator, self.physical_params),
            "robust": RobustMPC(self.config, self.scenario_generator, self.physical_params),
        }
        return self

    def build_case(self, date: tuple[int, int]) -> dict:
        """Prepare one experiment case for a month/day tuple."""
        if self.data is None:
            raise RuntimeError("ExperimentRunner.setup() must be called before build_case().")
        month, day = date
        return build_case_data(
            month,
            day,
            self.data["solar_data"],
            self.data["load_data"],
            self.config,
        )

    def run_single_case(self, date: tuple[int, int], method: str = "Uniform", beta: float | None = None):
        """Run the primary UA-TMPC case for one date."""
        return self.run_ua_tube_day(date, method=method, beta=beta)

    def run_standard_day(self, date: tuple[int, int]) -> dict:
        """Run the final_single.ipynb standard MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
                "standard"
            ].solve(case_data, price_data, curr_t, state["curr_ess"], state["curr_tau"])

            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, p_heat, p_cool = ther_ps
            tau = ther_taus[0]
            _, p_ch, p_dis = ess_ps
            ess = ess_esss[0]

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(p_heat, p_cool)
                + tau_error * params.gama1
            )
            curr_ess = state["curr_ess"] + params.eta_ch * p_ch - p_dis / params.eta_dch
            if elec_error >= 0:
                curr_ess += elec_error * params.eta_ch
            else:
                curr_ess += elec_error / params.eta_dch

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("standard", date, state, start_time)

    def run_perfect_day(self, date: tuple[int, int]) -> dict:
        """Run the final_single.ipynb perfect MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        real_data = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
            mpc_type="perfect",
        )
        real_temp, real_net_load = real_data
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
                "perfect"
            ].solve(real_data, price_data, curr_t, state["curr_ess"], state["curr_tau"])

            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, p_heat, p_cool = ther_ps
            tau = ther_taus[0]
            _, p_ch, p_dis = ess_ps
            ess = ess_esss[0]

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(p_heat, p_cool)
                + tau_error * params.gama1
            )
            curr_ess = state["curr_ess"] + params.eta_ch * p_ch - p_dis / params.eta_dch

            if elec_error >= 0:
                state["total_penalty"] += c_buy[curr_t] * elec_error
            else:
                state["total_penalty"] += -c_cur * elec_error

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("perfect", date, state, start_time)

    def run_ground_truth_day(self, date: tuple[int, int]) -> dict:
        """Run one 96-step optimization using the full day's ground-truth values."""
        self._require_setup()
        params = self.physical_params
        real_data = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
            "ground_truth"
        ].solve(real_data, price_data, 0, state["curr_ess"], state["curr_tau"])

        p_buy_profile, p_pvc_profile = controls
        _, p_heat_profile, p_cool_profile = ther_ps
        tau_profile = ther_taus[0]
        _, p_ch_profile, p_dis_profile = ess_ps
        ess_profile = ess_esss[0]

        for curr_t in range(24 * 4):
            p_buy = p_buy_profile[curr_t]
            p_pvc = p_pvc_profile[curr_t]
            p_heat = p_heat_profile[curr_t]
            p_cool = p_cool_profile[curr_t]
            tau = tau_profile[curr_t]
            p_ch = p_ch_profile[curr_t]
            p_dis = p_dis_profile[curr_t]
            ess = ess_profile[curr_t]

            state["before_ess_list"].append(ess)
            state["before_tau_list"].append(tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(ess, tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("ground_truth", date, state, start_time)

    def run_ua_tube_day(
        self,
        date: tuple[int, int],
        method: str = "Uniform",
        beta: float | None = None,
        soft: bool = False,
        ww: float | str = 1.0,
    ) -> dict:
        """Run the final_full.ipynb UA tube closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data
        if ww == "max":
            ww = float(np.max(c_buy))

        state = self._initial_day_state()
        state["alph_list"] = []
        state["ess_tube_lb"] = [params.ess_min]
        state["ess_tube_ub"] = [params.ess_max]
        state["tau_tube_lb"] = [params.tau_min[0]]
        state["tau_tube_ub"] = [params.tau_max[0]]

        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            if soft:
                result = self.optimizers["ua_tube"].solve_soft_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    ww=float(ww),
                    method=method,
                    beta=beta,
                )
            else:
                result = self.optimizers["ua_tube"].solve_hard_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    method=method,
                    beta=beta,
                )

            forecasts, controls, alphs, ther_ps, ther_taus, ess_ps, ess_esss = result
            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            alph_buy, alph_hvac, alph_ess, alph_pvc = alphs
            _, nom_p_heat, nom_p_cool, _, _ = ther_ps
            nom_tau, tau_ub, tau_lb = ther_taus
            _, nom_p_ch, nom_p_dis, _, _ = ess_ps
            nom_ess, ess_ub, ess_lb = ess_esss

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            hvac_error = alph_hvac * elec_error
            hvac_temp_effect = self._hvac_error_temp_effect(
                hvac_error,
                nom_p_heat,
                nom_p_cool,
                state["curr_tau"],
            )
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(nom_p_heat, nom_p_cool)
                + tau_error * params.gama1
                + hvac_temp_effect
            )
            ess_error = elec_error / params.eta_dch if elec_error < 0 else elec_error * params.eta_ch
            curr_ess = (
                state["curr_ess"]
                + params.eta_ch * nom_p_ch
                - nom_p_dis / params.eta_dch
                + alph_ess * ess_error
            )

            curr_p_buy = p_buy + alph_buy * elec_error
            curr_p_pvc = p_pvc - alph_pvc * elec_error

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(nom_ess)
            state["nom_tau_list"].append(nom_tau)
            state["alph_list"].append(alphs)
            state["ess_tube_lb"].append(ess_lb)
            state["ess_tube_ub"].append(ess_ub)
            state["tau_tube_lb"].append(tau_lb)
            state["tau_tube_ub"].append(tau_ub)
            state["cost_list"].append(
                c_buy[curr_t] * curr_p_buy
                + params.c_deg * (nom_p_ch * params.eta_ch + nom_p_dis / params.eta_dch)
                + c_cur * curr_p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        result_method = "soft_ua_tube" if soft else "hard_ua_tube"
        return self._finalize_day_result(result_method, date, state, start_time)

    def run_tube_day(
        self,
        date: tuple[int, int],
        method: str = "Uniform",
        beta: float | None = None,
        soft: bool = False,
        ww: float | str = 1.0,
        alphas: tuple[
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
        ]
        | None = None,
        optimizer_name: str = "tube",
        result_method: str | None = None,
    ) -> dict:
        """Run fixed-alpha tube MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data
        if ww == "max":
            ww = float(np.max(c_buy))

        if not soft and optimizer_name == "stochastic_tmpc":
            raise ValueError("Stochastic TMPC only supports soft tube constraints.")

        optimizer = self.optimizers[optimizer_name]
        if alphas is not None:
            optimizer_cls = StochasticTMPC if optimizer_name == "stochastic_tmpc" else TubeMPC
            optimizer = optimizer_cls(self.config, self.scenario_generator, self.physical_params, alphas=alphas)

        state = self._initial_day_state()
        state["alph_list"] = []
        state["ess_tube_lb"] = [params.ess_min]
        state["ess_tube_ub"] = [params.ess_max]
        state["tau_tube_lb"] = [params.tau_min[0]]
        state["tau_tube_ub"] = [params.tau_max[0]]

        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            if soft:
                result = optimizer.solve_soft_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    ww=float(ww),
                    method=method,
                    beta=beta,
                )
            else:
                result = optimizer.solve_hard_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    method=method,
                    beta=beta,
                )

            forecasts, controls, fixed_alphas, ther_ps, ther_taus, ess_ps, ess_esss = result
            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            alph_buy, alph_hvac, alph_ess, alph_pvc = fixed_alphas
            _, nom_p_heat, nom_p_cool, _, _ = ther_ps
            nom_tau, tau_ub, tau_lb = ther_taus
            _, nom_p_ch, nom_p_dis, _, _ = ess_ps
            nom_ess, ess_ub, ess_lb = ess_esss

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            hvac_error = alph_hvac * elec_error
            hvac_temp_effect = self._hvac_error_temp_effect(
                hvac_error,
                nom_p_heat,
                nom_p_cool,
                state["curr_tau"],
            )
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(nom_p_heat, nom_p_cool)
                + tau_error * params.gama1
                + hvac_temp_effect
            )
            ess_error = elec_error / params.eta_dch if elec_error < 0 else elec_error * params.eta_ch
            curr_ess = (
                state["curr_ess"]
                + params.eta_ch * nom_p_ch
                - nom_p_dis / params.eta_dch
                + alph_ess * ess_error
            )

            curr_p_buy = p_buy + alph_buy * elec_error
            curr_p_pvc = p_pvc - alph_pvc * elec_error

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(nom_ess)
            state["nom_tau_list"].append(nom_tau)
            state["alph_list"].append(fixed_alphas)
            state["ess_tube_lb"].append(ess_lb)
            state["ess_tube_ub"].append(ess_ub)
            state["tau_tube_lb"].append(tau_lb)
            state["tau_tube_ub"].append(tau_ub)
            state["cost_list"].append(
                c_buy[curr_t] * curr_p_buy
                + params.c_deg * (nom_p_ch * params.eta_ch + nom_p_dis / params.eta_dch)
                + c_cur * curr_p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        if result_method is None:
            result_method = "soft_tube" if soft else "hard_tube"
        return self._finalize_day_result(result_method, date, state, start_time)

    def run_stochastic_tmpc_day(
        self,
        date: tuple[int, int],
        method: str = "Uniform",
        beta: float | None = None,
        ww: float | str = 1.0,
        alphas: tuple[
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
            float | np.ndarray,
        ]
        | None = None,
    ) -> dict:
        """Run stochastic TMPC closed-loop simulation for one day."""
        return self.run_tube_day(
            date,
            method=method,
            beta=beta,
            soft=True,
            ww=ww,
            alphas=alphas,
            optimizer_name="stochastic_tmpc",
            result_method="stochastic_tmpc",
        )

    def run_wdr_day(
        self,
        date: tuple[int, int],
        method: str = "Uniform",
        beta: float | None = None,
        soft: bool = False,
        ww: float | str = 1.0,
    ) -> dict:
        """Run distributionally robust tube MPC for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data
        if ww == "max":
            ww = float(np.max(c_buy))

        state = self._initial_day_state()
        state["ess_tube_lb"] = [params.ess_min]
        state["ess_tube_ub"] = [params.ess_max]
        state["tau_tube_lb"] = [params.tau_min[0]]
        state["tau_tube_ub"] = [params.tau_max[0]]

        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            if soft:
                result = self.optimizers["wdr"].solve_soft_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    ww=float(ww),
                    method=method,
                    beta=beta,
                )
            else:
                result = self.optimizers["wdr"].solve_hard_tube(
                    case_data,
                    price_data,
                    curr_t,
                    state["curr_ess"],
                    state["curr_tau"],
                    method=method,
                    beta=beta,
                )

            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = result
            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, nom_p_heat, nom_p_cool, _, _ = ther_ps
            nom_tau, tau_ub, tau_lb = ther_taus
            _, nom_p_ch, nom_p_dis, _, _ = ess_ps
            nom_ess, ess_ub, ess_lb = ess_esss

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(nom_p_heat, nom_p_cool)
                + tau_error * params.gama1
            )
            
            curr_ess = (
                state["curr_ess"]
                + params.eta_ch * nom_p_ch
                - nom_p_dis / params.eta_dch
            )
            if elec_error >= 0:
                curr_ess += elec_error * params.eta_ch
            else:
                curr_ess += elec_error / params.eta_dch

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(nom_ess)
            state["nom_tau_list"].append(nom_tau)
            state["ess_tube_lb"].append(ess_lb)
            state["ess_tube_ub"].append(ess_ub)
            state["tau_tube_lb"].append(tau_lb)
            state["tau_tube_ub"].append(tau_ub)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (nom_p_ch * params.eta_ch + nom_p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        result_method = "soft_wdr" if soft else "hard_wdr"
        return self._finalize_day_result(result_method, date, state, start_time)

    def run_scenario_day(self, date: tuple[int, int], method: str = "Uniform") -> dict:
        """Run the final_single.ipynb scenario MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
                "scenario"
            ].solve(case_data, price_data, curr_t, state["curr_ess"], state["curr_tau"], method=method)

            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, p_heat, p_cool = ther_ps
            tau = ther_taus[0]
            _, p_ch, p_dis = ess_ps
            ess = ess_esss[0]

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(p_heat, p_cool)
                + tau_error * params.gama1
            )
            curr_ess = state["curr_ess"] + params.eta_ch * p_ch - p_dis / params.eta_dch
            if elec_error >= 0:
                curr_ess += elec_error * params.eta_ch
            else:
                curr_ess += elec_error / params.eta_dch

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("scenario", date, state, start_time)

    def run_dr_day(
        self,
        date: tuple[int, int],
        method: str = "Uniform",
        beta: float | None = None,
    ) -> dict:
        """Run the distributionally robust MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
                "dr"
            ].solve(
                case_data,
                price_data,
                curr_t,
                state["curr_ess"],
                state["curr_tau"],
                method=method,
                beta=beta,
            )

            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, p_heat, p_cool = ther_ps
            tau = ther_taus[0]
            _, p_ch, p_dis = ess_ps
            ess = ess_esss[0]

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(p_heat, p_cool)
                + tau_error * params.gama1
            )
            curr_ess = state["curr_ess"] + params.eta_ch * p_ch - p_dis / params.eta_dch
            if elec_error >= 0:
                curr_ess += elec_error * params.eta_ch
            else:
                curr_ess += elec_error / params.eta_dch

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("dr", date, state, start_time)

    def run_robust_day(self, date: tuple[int, int], spare: float = 0.25) -> dict:
        """Run the final_single.ipynb robust MPC closed-loop simulation for one day."""
        self._require_setup()
        params = self.physical_params
        case_data = self.build_case(date)
        real_temp, real_net_load = build_data_y_window(
            date[0],
            date[1],
            self.data["solar_data"],
            self.data["load_data"],
        )
        price_data = elec_price_read(date[0], date[1], self.data["da_price"], self.data["rt_price"])
        c_buy, c_cur, c_pen = price_data

        state = self._initial_day_state()
        start_time = time.time()
        for curr_t in range(0, 24 * 4):
            forecasts, controls, ther_ps, ther_taus, ess_ps, ess_esss = self.optimizers[
                "robust"
            ].solve(case_data, price_data, curr_t, state["curr_ess"], state["curr_tau"], spare=spare)

            forecast_net_load, forecast_tau = forecasts
            p_buy, p_pvc = controls
            _, p_heat, p_cool = ther_ps
            tau = ther_taus[0]
            _, p_ch, p_dis = ess_ps
            ess = ess_esss[0]

            elec_error = (real_net_load[curr_t] - forecast_net_load) * params.www
            tau_error = (real_temp[curr_t] - forecast_tau) * params.www
            curr_tau = (
                state["curr_tau"]
                + params.gama1 * (forecast_tau - state["curr_tau"])
                + params.thermal_power_effect(p_heat, p_cool)
                + tau_error * params.gama1
            )
            curr_ess = state["curr_ess"] + params.eta_ch * p_ch - p_dis / params.eta_dch
            if elec_error >= 0:
                curr_ess += elec_error * params.eta_ch
            else:
                curr_ess += elec_error / params.eta_dch

            state["before_ess_list"].append(curr_ess)
            state["before_tau_list"].append(curr_tau)
            curr_ess, curr_tau, penalty = self._apply_state_correction(curr_ess, curr_tau, c_cur, c_pen, curr_t)
            state["total_penalty"] += penalty

            state["ess_list"].append(curr_ess)
            state["tau_list"].append(curr_tau)
            state["nom_ess_list"].append(ess)
            state["nom_tau_list"].append(tau)
            state["cost_list"].append(
                c_buy[curr_t] * p_buy
                + params.c_deg * (p_ch * params.eta_ch + p_dis / params.eta_dch)
                + c_cur * p_pvc
                + params.c_ther * (curr_tau - params.tau_ref) ** 2
            )
            state["curr_ess"] = curr_ess
            state["curr_tau"] = curr_tau

        return self._finalize_day_result("robust", date, state, start_time)

    def run_comparison(self, date: tuple[int, int]) -> dict:
        """Run all benchmark methods for one date."""
        x_data = self.build_case(date)
        raise NotImplementedError(
            "Connect this method after all optimizer implementations are migrated from the notebooks."
        )

    def _require_setup(self) -> None:
        if self.data is None:
            raise RuntimeError("ExperimentRunner.setup() must be called before running experiments.")

    def _initial_day_state(self) -> dict:
        params = self.physical_params
        return {
            "ess_list": [params.ess_int],
            "nom_ess_list": [params.ess_int],
            "tau_list": [params.tau_ini],
            "nom_tau_list": [params.tau_ini],
            "before_tau_list": [params.tau_ini],
            "before_ess_list": [params.ess_int],
            "cost_list": [],
            "curr_ess": params.ess_int,
            "curr_tau": params.tau_ini,
            "total_penalty": 0,
        }

    def _apply_state_correction(self, curr_ess, curr_tau, c_cur, c_pen, curr_t: int):
        params = self.physical_params
        penalty = 0

        if curr_ess >= params.ess_max:
            penalty += c_cur * (curr_ess - params.ess_max)
            curr_ess = params.ess_max
        elif curr_ess < params.ess_min:
            penalty += c_pen[curr_t] * (params.ess_min - curr_ess)
            curr_ess = params.ess_min
        else:
            curr_ess = np.asarray(curr_ess).item()

        if curr_tau >= params.tau_max[curr_t]:
            penalty += c_pen[curr_t] * (curr_tau - params.tau_max[curr_t]) / (
                params.gama2 * params.eta_cool
            )
            curr_tau = params.tau_max[curr_t]
        elif curr_tau < params.tau_min[curr_t]:
            penalty += c_pen[curr_t] * (params.tau_min[curr_t] - curr_tau) / (
                params.gama2 * params.eta_heat
            )
            curr_tau = params.tau_min[curr_t]
        else:
            curr_tau = np.asarray(curr_tau).item()

        return curr_ess, curr_tau, penalty

    def _hvac_error_temp_effect(self, hvac_error, p_heat, p_cool, curr_tau):
        params = self.physical_params
        hvac_error = np.asarray(hvac_error).item()
        p_heat = np.asarray(p_heat).item()
        p_cool = np.asarray(p_cool).item()
        curr_tau = np.asarray(curr_tau).item()

        if p_heat > 1e-6:
            return -params.gama2 * params.eta_heat * hvac_error
        if p_cool > 1e-6:
            return params.gama2 * params.eta_cool * hvac_error
        if curr_tau < params.tau_ref:  # used for the case where p_heat and p_cool all equal to 0
            return -params.gama2 * params.eta_heat * hvac_error
        return params.gama2 * params.eta_cool * hvac_error

    def _finalize_day_result(self, method: str, date: tuple[int, int], state: dict, start_time: float) -> dict:
        operational_cost = np.sum(state["cost_list"])
        total_penalty = state["total_penalty"]
        total_cost = operational_cost + total_penalty
        result = {
            "method": method,
            "date": date,
            "runtime": time.time() - start_time,
            "total_cost": np.asarray(total_cost).item(),
            "operational_cost": np.asarray(operational_cost).item(),
            "total_penalty": np.asarray(total_penalty).item(),
            "ess_list": state["ess_list"],
            "nom_ess_list": state["nom_ess_list"],
            "tau_list": state["tau_list"],
            "nom_tau_list": state["nom_tau_list"],
            "before_ess_list": state["before_ess_list"],
            "before_tau_list": state["before_tau_list"],
            "cost_list": state["cost_list"],
        }
        for optional_key in (
            "alph_list",
            "ess_tube_lb",
            "ess_tube_ub",
            "tau_tube_lb",
            "tau_tube_ub",
        ):
            if optional_key in state:
                result[optional_key] = state[optional_key]
        return result
