"""Scenario generation utilities."""

from __future__ import annotations

import numpy as np

from .config import MPCConfig
from .forecasting import enforce_non_crossing


def dict_to_array(preds: dict, horizon: int, quantiles) -> np.ndarray:
    """Convert nested quantile predictions into a horizon-by-quantile array."""
    array = np.zeros((horizon, len(quantiles)))
    for i in range(1, horizon + 1):
        for j, q in enumerate(quantiles):
            array[i - 1, j] = preds[i].get(q, np.nan)
    return array


def scenario_weights(temp_scenarios, electricity_scenarios, method: str = "Uniform") -> list[float]:
    """
        Assign scenario weights.
        Different weighting methods can be implemented later.
    """
    n_scenarios = temp_scenarios.shape[0]
    if method == "Uniform":
        return [1 / n_scenarios for _ in range(n_scenarios)]
    raise NotImplementedError(f"Scenario weighting method '{method}' is not implemented yet.")


class ScenarioGenerator:
    """Generate temperature, PV, load, and electricity scenarios for MPC."""

    def __init__(self, temp_model, pv_model, load_model, config: MPCConfig):
        self.temp_model = temp_model
        self.pv_model = pv_model
        self.load_model = load_model
        self.config = config

    def generate(self, x_data: dict | tuple, t: int, method: str = "Uniform"):
        """Generate forecast means and centered scenarios at one MPC time step."""
        rng = np.random.RandomState(self.config.random_seed)
        quantiles = self.config.quantiles
        horizon = self.config.control_horizon
        n_scenarios = self.config.n_scenarios

        if isinstance(x_data, dict):
            x_temp = x_data["x_temp"]
            x_pv = x_data["x_pv"]
            x_load = x_data["x_load"]
        else:
            x_temp, x_pv, x_load = x_data

        curr_x_temp = x_temp[t : t + 1, :]
        curr_x_pv = x_pv[t : t + 1, :]
        curr_x_load = x_load[t : t + 1, :]

        temp_qs = enforce_non_crossing(self.temp_model.predict(curr_x_temp, quantiles), quantiles)
        temp_qs = dict_to_array(temp_qs, horizon, quantiles)
        scenario_by_step = [np.zeros((3, n_scenarios)) for _ in range(horizon)]

        for n in range(n_scenarios):
            current_temp = np.zeros((1, horizon))
            for step in range(horizon):
                current_temp[0, step] = temp_qs[step, rng.randint(len(quantiles))]

            pv_qs = dict_to_array(
                enforce_non_crossing(
                    self.pv_model.predict(np.hstack([curr_x_pv, current_temp]), quantiles),
                    quantiles,
                ),
                horizon,
                quantiles,
            )
            load_qs = dict_to_array(
                enforce_non_crossing(
                    self.load_model.predict(np.hstack([curr_x_load, current_temp]), quantiles),
                    quantiles,
                ),
                horizon,
                quantiles,
            )
            current_pv = np.zeros((1, horizon))
            current_load = np.zeros((1, horizon))
            for step in range(horizon):
                current_pv[0, step] = pv_qs[step, rng.randint(len(quantiles))]
                current_load[0, step] = load_qs[step, rng.randint(len(quantiles))]

            for step in range(horizon):
                scenario_by_step[step][0, n] = current_temp[0, step]
                scenario_by_step[step][1, n] = current_pv[0, step]
                scenario_by_step[step][2, n] = current_load[0, step]

        stacked = np.stack(scenario_by_step, axis=0)
        scenarios = stacked.transpose(2, 0, 1)
        scenarios[:, :, 2] = scenarios[:, :, 2] / 5e3

        point_forecast = np.mean(scenarios, axis=0)
        scenarios = scenarios - point_forecast

        final_temp_forecast = point_forecast[:, 0]
        final_elec_forecast = point_forecast[:, 2] - point_forecast[:, 1]
        final_temp_scenarios = scenarios[:, :, 0]
        final_elec_scenarios = scenarios[:, :, 2] - scenarios[:, :, 1]
        weights = scenario_weights(final_temp_scenarios, final_temp_scenarios, method="Uniform")

        return (
            final_temp_forecast,
            final_elec_forecast,
            final_temp_scenarios,
            final_elec_scenarios,
            weights,
        )
