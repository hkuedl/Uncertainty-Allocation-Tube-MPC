"""Data loading and feature-window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MPCConfig, Paths


MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
CUMULATIVE_DAYS_BEFORE_MONTH = np.insert(np.cumsum(MONTH_DAYS), 0, 0)


def load_solar_data(path) -> pd.DataFrame:
    """Read weather/PV data and add cyclic time features used by forecasts."""
    data = pd.read_csv(path, encoding="gbk")
    data["hour"] = np.sin(2 * np.pi * (data["hour"] * 60 + data["min"]) / (24 * 60))
    months = data["month"].to_numpy(dtype=int)
    days = data["day"].to_numpy(dtype=int)
    data["doy"] = np.sin((CUMULATIVE_DAYS_BEFORE_MONTH[months - 1] + days) * 2 * np.pi / 365)
    return data


def load_building_load(path) -> pd.DataFrame:
    """Read building load data."""
    return pd.read_csv(path)


def load_price_data(path, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read UK electricity price data and build DA/RT price arrays."""
    from scipy import interpolate

    price_data = pd.read_csv(path, encoding="gbk")
    price_data["Datetime (Local)"] = pd.to_datetime(price_data["Datetime (Local)"])
    df_2023 = price_data[price_data["Datetime (Local)"].dt.year == 2023]
    elec_price = df_2023["Price (EUR/MWhe)"].to_numpy() * 7.8 / 1000

    x_old = np.arange(8760)
    x_new = np.linspace(0, 8759, 8760 * 4)
    f_linear = interpolate.interp1d(x_old, elec_price, kind="linear")
    elec_price = f_linear(x_new)
    # To avoid non-positive electricity price
    elec_price = elec_price - np.min(elec_price) + 0.01
    rng = np.random.default_rng(seed)
    da_price = elec_price.copy()
    # To randomly generate the real-time price
    rt_price = elec_price * rng.uniform(1.2, 1.5, size=len(elec_price))
    return da_price, rt_price


def build_x_windows(df: pd.DataFrame, x_features, y_features, ctx: int = 24 * 4) -> np.ndarray:
    """Build historical feature windows for quantile forecasting."""
    base_features = list(y_features) + list(x_features)
    arr = df[base_features].values
    x_windows = []
    for i in range(ctx, len(df)):
        x_windows.append(arr[i - ctx : i, :].reshape(-1))
    return np.stack(x_windows)


def build_data_x_window(
    candidate_month: int,
    candidate_day: int,
    solar_data: pd.DataFrame,
    load_data: pd.DataFrame,
    config: MPCConfig,
) -> list[np.ndarray]:
    """Create temperature, PV, and load feature windows for one candidate date."""
    start = (CUMULATIVE_DAYS_BEFORE_MONTH[candidate_month - 1] + candidate_day - 2) * 24 * 4
    end = start + 2 * 4 * 24
    mask = (solar_data["month"] == candidate_month) & solar_data["day"].isin(
        [candidate_day, candidate_day - 1]
    )
    selected_solar = solar_data.loc[mask].copy()
    selected_load = load_data.iloc[start:end, :].copy()
    selected_load["outdoor temperature"] = selected_solar["Temp"].values

    x_temp = build_x_windows(
        selected_solar,
        ["hour", "Irradiance", "Rainfall", "RH", "SLP", "WS", "WD"],
        ["Temp"],
        ctx=config.context_steps,
    )
    x_pv = build_x_windows(
        selected_solar,
        ["hour", "Irradiance", "Rainfall", "RH", "SLP", "Temp", "Vis", "WS", "WD", "doy"],
        ["generation"],
        ctx=config.context_steps,
    )
    x_load = build_x_windows(
        selected_load,
        ["outdoor temperature", "time"],
        ["load"],
        ctx=config.context_steps,
    )
    return [x_temp, x_pv, x_load]


def build_case_data(
    candidate_month: int,
    candidate_day: int,
    solar_data: pd.DataFrame,
    load_data: pd.DataFrame,
    config: MPCConfig,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Prepare forecast windows and selected raw data for one candidate date."""
    start = (CUMULATIVE_DAYS_BEFORE_MONTH[candidate_month - 1] + candidate_day - 2) * 24 * 4
    end = start + 2 * 4 * 24
    mask = (solar_data["month"] == candidate_month) & solar_data["day"].isin(
        [candidate_day - 1, candidate_day]
    )
    selected_solar = solar_data.loc[mask].copy()
    selected_load = load_data.iloc[start:end, :].copy()
    selected_load["outdoor temperature"] = selected_solar["Temp"].values
    x_temp, x_pv, x_load = build_data_x_window(
        candidate_month,
        candidate_day,
        solar_data,
        load_data,
        config,
    )
    return {
        "selected_solar": selected_solar,
        "selected_load": selected_load,
        "x_temp": x_temp,
        "x_pv": x_pv,
        "x_load": x_load,
    }


def build_data_y_window(
    candidate_month: int,
    candidate_day: int,
    solar_data: pd.DataFrame,
    load_data: pd.DataFrame,
    mpc_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create real temperature and real net-load windows used by simulations."""
    y_temp_features = ["Temp"]
    y_pv_features = ["generation"]
    y_load_features = ["load"]
    start = (CUMULATIVE_DAYS_BEFORE_MONTH[candidate_month - 1] + candidate_day - 2) * 24 * 4

    if mpc_type == "perfect":
        end = start + 48 * 4 + 4 * 4
        selected_solar = solar_data.iloc[start:end, :]
        selected_load = load_data.iloc[start:end, :]
        real_temp = selected_solar[y_temp_features].iloc[-24 * 4 - 4 * 4 :].to_numpy()
        real_pv = selected_solar[y_pv_features].iloc[-24 * 4 - 4 * 4 :].to_numpy()
        # 5e3 is the scaling factor
        real_load = selected_load[y_load_features].iloc[-24 * 4 - 4 * 4 :].to_numpy() / 5e3
    else:
        end = start + 48 * 4
        mask = (solar_data["month"] == candidate_month) & solar_data["day"].isin(
            [candidate_day - 1, candidate_day]
        )
        selected_solar = solar_data.loc[mask]
        selected_load = load_data.iloc[start:end, :]
        real_temp = selected_solar[y_temp_features].iloc[-24 * 4 :].to_numpy()
        real_pv = selected_solar[y_pv_features].iloc[-24 * 4 :].to_numpy()
        real_load = selected_load[y_load_features].iloc[-24 * 4 :].to_numpy() / 5e3

    real_net_load = real_load - real_pv
    return real_temp, real_net_load


def elec_price_read(
    candidate_month: int,
    candidate_day: int,
    da_price: np.ndarray,
    rt_price: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Read the buy, curtailment, and penalty prices for one candidate day."""
    start = (CUMULATIVE_DAYS_BEFORE_MONTH[candidate_month - 1] + candidate_day - 1) * 4 * 24
    end = start + 4 * 24 + 4 * 2
    c_buy = da_price[start:end]
    c_cur = 1.0
    c_pen = rt_price[start:end]
    return c_buy, c_cur, c_pen


def load_all_data(paths: Paths, config: MPCConfig) -> dict[str, object]:
    """Load all raw data sources used by the experiment runner."""
    solar_data = load_solar_data(paths.solar_data)
    load_data = load_building_load(paths.load_data)
    da_price, rt_price = load_price_data(paths.price_data, seed=config.random_seed)
    return {
        "solar_data": solar_data,
        "load_data": load_data,
        "da_price": da_price,
        "rt_price": rt_price,
    }
