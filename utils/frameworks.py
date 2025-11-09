from utils.config import Config
from typing import Any, Dict, List
import PSID.PSID as PSID
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np


class BaseFramework:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = get_logger()

    def _initalize_model(self):
        raise NotImplemented

    def _train(self, Y, Z=None):
        self.logger.info(f"Initializing model and starting training.")
        self.model = self._initalize_model()
        self.logger.info(f"Model initialized: {self.model}")
        return self.model.train(Y, Z)

    def _validate(self, Y):
        self.logger.info("Starting validation...")
        return self.model.validate(Y)

    def _test(self, Y):
        self.logger.info("Starting test...")
        return self.model.test(Y)

    def _predict(self, Y):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y)


class PSIDWrapper:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def train(self, Y, Z=None):

        nx = self.config.model.nx
        n1 = self.config.model.n1
        i = self.config.model.i
        time_first = self.config.model.time_first
        remove_mean_Y = self.config.model.remove_mean_Y
        remove_mean_Z = self.config.model.remove_mean_Z
        zscore_Y = self.config.model.zscore_Y
        zscore_Z = self.config.model.zscore_Z

        self.logger.info(
            f"Calling PSID.PSID with nx={nx}, n1={n1}, i={i}, time_first={time_first}; "
        )

        self.idSys = PSID(
            Y,
            Z,
            nx,
            n1,
            i,
            time_first=time_first,
            remove_mean_Y=remove_mean_Y,
            remove_mean_Z=remove_mean_Z,
            zscore_Y=zscore_Y,
            zscore_Z=zscore_Z,
        )
        return self.idSys

    def predict(self, Y):
        return self.idSys.predict(Y)

    def validate(self, Y):
        Zp, Yp, Xp = self.idSys.predict(Y)

        r_list, r_mean = pearson_r_per_channel(Y, Yp)
        result = {
            "Y": Y,
            "Zp": Zp,
            "Yp": Yp,
            "Xp": Xp,
            "Yp_shape": state_shape(Yp),
            "Zp_shape": (None if Zp is None else state_shape(Zp)),
            "Xp_shape": (None if Xp is None else state_shape(Xp)),
            "pearson_r_per_channel": r_list,  # per trial if multiple trials, else per-channel list
            "pearson_r_mean": r_mean,  # overall mean across trials/channels
        }

        return result

    def test(self, Y):
        return self.validate(Y)

    def forecast(self, m, Y_past, U_past=None, U_future=None):
        return self.idSys.forecast(m, Y_past, U_past=U_past, U_future=U_future)

    # Backward-compatible alias for misspelled calls
    def forcast(self, m, Y_past, U_past=None, U_future=None):
        return self.forecast(m, Y_past, U_past=U_past, U_future=U_future)

    def validate_forecast(self, Y_list, margin=None):
        """Validate m-step ahead forecast using config.model.forcast.m, with optional margin before the end.

        We forecast m steps not at the very end, but ending `margin` seconds before
        the end of each trial. Sampling frequency is assumed to be 1000 Hz as requested.

        For each trial y (length T), define margin_samples = int(1000 * margin_sec).
        The forecast window is y[ T-(m+margin_samples) : T-margin_samples ]. The model
        is initialized with the past y[: T-(m+margin_samples)].

        Args:
            Y_list: list of trials (each T x ny ndarray) or a single ndarray.
            margin: None, float seconds, or list of float seconds per trial. If None or 0, uses last m steps.

        Returns a dict with per-trial arrays and metrics to facilitate plotting.
        """
        m = self.config.model.forcast.m

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],  # true past + predicted future up to T - margin_samples
            "Z_future_pred": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
        }

        # Ensure list input
        if not isinstance(Y_list, (list, tuple)):
            Y_list = [Y_list]

        # Normalize margin to per-trial list (in seconds)
        if margin is None or (
            isinstance(margin, (int, float)) and float(margin) == 0.0
        ):
            margin_list = [0.0] * len(Y_list)
        elif isinstance(margin, (int, float)):
            margin_list = [float(margin)] * len(Y_list)
        else:
            margin_list = list(margin)
            if len(margin_list) != len(Y_list):
                # If mismatch, fallback to zeros
                margin_list = [0.0] * len(Y_list)

        # Iterate trials
        for idx, Y in enumerate(Y_list):
            if Y is None or len(Y) == 0:
                # Skip empty entries
                results["Y_future_true"].append(None)
                results["Y_future_pred"].append(None)
                results["Y_concat_for_plot"].append(None)
                results["Z_future_pred"].append(None)
                results["X_future_pred"].append(None)
                results["pearson_per_channel"].append([])
                continue

            T = Y.shape[0]
            if m >= T:
                raise ValueError(
                    f"Forecast horizon m={m} must be smaller than trial length T={T}"
                )

            margin_sec = float(margin_list[idx]) if margin_list is not None else 0.0
            margin_samples = int(1000 * margin_sec)

            # Compute indices: [-(m+margin_samples) : -margin_samples]
            start = T - (m + margin_samples)
            end = T - margin_samples
            self.logger.info(
                f"Got {start}:{end} forecast window for trial {idx} with T={T}, m={m}, margin_sec={margin_sec} (samples={margin_samples})"
            )
            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid margin for trial {idx}: margin_sec={margin_sec} (samples={margin_samples}), T={T}, m={m}"
                )

            Y_past = Y[margin_samples:start]
            Y_future_true = Y[start:end]

            Zf, Yf, Xf = self.forecast(m, Y_past)

            if Yf is None:
                raise RuntimeError("Model returned no Y forecast")

            # Build concatenated series for plotting: true past + predicted future (excludes final margin)
            Y_concat = np.concatenate([Y_past, Yf], axis=0)

            # Pearson r per channel between true future and predicted future
            r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
            r_list = (
                r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else r_list
            )

            results["Y_future_true"].append(Y_future_true)
            results["margin_samples"] = margin_samples
            results["Y_future_pred"].append(Yf)
            results["Y_concat_for_plot"].append(Y_concat)
            results["Z_future_pred"].append(Zf)
            results["X_future_pred"].append(Xf)
            results["pearson_per_channel"].append(r_list)

        flat_r = []
        for r in results["pearson_per_channel"]:
            if r is None:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r.append(float(v))
        results["pearson_overall_mean"] = (
            float(np.mean(flat_r)) if len(flat_r) > 0 else np.nan
        )

        return results


class PSIDFramework(BaseFramework):
    def _initalize_model(self):
        self.logger.info("Initializing PSIDAdapter (function-based PSID API)")
        return PSIDWrapper(self.config)
