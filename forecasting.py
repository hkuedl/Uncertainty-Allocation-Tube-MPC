"""Forecast model loading and quantile prediction."""

from __future__ import annotations

import json
from pathlib import Path


def enforce_non_crossing(preds: dict, quantiles) -> dict:
    """Ensure predicted quantiles are monotone non-decreasing."""
    for horizon in sorted(preds.keys()):
        vals = [preds[horizon][q] for q in quantiles]
        monotone = []
        current = vals[0]
        monotone.append(current)
        for value in vals[1:]:
            current = max(current, value)
            monotone.append(current)
        for i, q in enumerate(quantiles):
            preds[horizon][q] = monotone[i]
    return preds


class QuantileForecaster:
    """XGBoost quantile forecaster loaded from a manifest directory."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.boosters = []
        self.manifest = None

    def load(self) -> "QuantileForecaster":
        import xgboost as xgb

        manifest_path = self.model_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)

        quantiles = manifest["quantiles"]
        boosters = []
        for hi, _ in enumerate(manifest["horizons"]):
            row = []
            for qi, _ in enumerate(quantiles):
                model_info = manifest["models"][hi * len(quantiles) + qi]
                model_path = Path(model_info["path"])
                if not model_path.is_absolute():
                    model_path = self.model_dir.parent / model_path
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                row.append(booster)
            boosters.append(row)

        self.manifest = manifest
        self.boosters = boosters
        return self

    def predict(self, x_new, quantiles) -> dict:
        import xgboost as xgb

        if not self.boosters:
            raise RuntimeError("QuantileForecaster.load() must be called before predict().")

        dtest = xgb.DMatrix(x_new)
        results = {}
        for hi, horizon_boosters in enumerate(self.boosters):
            horizon = hi + 1
            row = {}
            for qi, q in enumerate(quantiles):
                preds = horizon_boosters[qi].predict(dtest)
                row[q] = float(preds.ravel()[0])
            results[horizon] = row
        return results
