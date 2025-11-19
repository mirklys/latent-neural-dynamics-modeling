from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np
import sys
from pathlib import Path

# Add DPAD to path
dpad_path = Path(__file__).parent.parent / "DPAD-main" / "source"
if str(dpad_path) not in sys.path:
    sys.path.insert(0, str(dpad_path))

# Simple type aliases - always lists of trials
Array2D = NDArray[np.float64]
TrialList = List[Array2D]  # List of (time, channels) arrays


class BaseFramework:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.logger = get_logger()

    def _initialize_model(self):
        raise NotImplementedError

    def _train(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Initializing model and starting training.")
        self.model = self._initialize_model()
        self.logger.info(f"Model initialized: {self.model}")
        return self.model.train(Y, Z)

    def _validate(self, Y: TrialList) -> Dict[str, Any]:
        self.logger.info("Starting validation...")
        return self.model.validate(Y)

    def _test(self, Y: TrialList) -> Dict[str, Any]:
        self.logger.info("Starting test...")
        return self.model.test(Y)

    def _predict(self, Y: TrialList):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y)

    def _forecast(self, m: int, Y_past: Array2D):
        self.logger.info(f"Running {m}-step ahead forecast...")
        return self.model.forecast(m, Y_past)

    def _validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.logger.info("Starting forecast validation...")
        return self.model.validate_forecast(Y_list, Z_list=Z_list, margin=margin)


class PSIDWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from PSID.PSID import PSID as PSIDClass

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        i: int = self.config.model.i
        time_first: bool = self.config.model.time_first
        remove_mean_Y: bool = self.config.model.remove_mean_Y
        remove_mean_Z: bool = self.config.model.remove_mean_Z
        zscore_Y: bool = self.config.model.zscore_Y
        zscore_Z: bool = self.config.model.zscore_Z

        self.logger.info(
            f"Calling PSID.PSID with nx={nx}, n1={n1}, i={i}, time_first={time_first}; "
        )

        self.idSys = PSIDClass(
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

    def predict(self, Y: TrialList):
        return self.idSys.predict(Y)

    def validate(self, Y: TrialList) -> Dict[str, Any]:
        Zp, Yp, Xp = self.idSys.predict(Y)
        r_list, r_mean = pearson_r_per_channel(Y, Yp if Yp is not None else Y)
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

    def test(self, Y: TrialList) -> Dict[str, Any]:
        return self.validate(Y)

    def forecast(self, m: int, Y_past: Array2D):
        return self.idSys.forecast(m, Y_past)

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate m-step ahead forecast using config.model.forecast.m, with optional margin before the end.

        We forecast m steps not at the very end, but ending `margin` seconds before
        the end of each trial. Note: Y_list and Z_list should already have margins removed
        (as done by _slice_data), so we don't remove them again here.

        For each trial y (length T), the forecast window is y[T-m:T].
        The model is initialized with the past y[:T-m].

        Args:
            Y_list: list of trials (each T x ny ndarray), margins already removed.
            Z_list: optional list of behavioral trials (each T x nz ndarray), margins already removed.
            margin: Not used anymore since margins are pre-removed, kept for backward compatibility.

        Returns a dict with per-trial arrays and metrics to facilitate plotting.
        """
        m = self.config.model.forecast.m
        margin_sec = margin if margin is not None else 0.0

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
        }

        # Iterate trials
        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if m >= T:
                raise ValueError(
                    f"Forecast horizon m={m} must be smaller than trial length T={T}"
                )

            # Forecast the last m steps (margins already removed from Y_list)
            start = T - m
            end = T

            # Use all available past data (margins already removed)
            Y_past = Y[:start]
            Y_future_true = Y[start:end]

            # Get corresponding Z data if available
            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[start:end] if Z is not None else None

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

            # Compute Z correlations if both true and predicted Z are available
            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = (
                    r_list_Z[0]
                    if isinstance(r_list_Z, list) and len(r_list_Z) > 0
                    else r_list_Z
                )
            else:
                r_list_Z = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist())
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(
                Z_future_true.tolist() if Z_future_true is not None else None
            )
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        # Compute overall means for Y
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

        # Compute overall means for Z
        flat_r_Z = []
        for r in results["pearson_per_channel_Z"]:
            if r is None or not r:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r_Z.append(float(v))
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if len(flat_r_Z) > 0 else np.nan
        )

        return results


class PSIDFramework(BaseFramework):
    def _initialize_model(self):
        return PSIDWrapper(self.config)


class DPADWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from DPAD import DPADModel

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        method_code: str = self.config.model.method_code
        epochs: int = self.config.model.epochs

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}"
        )
        # DPAD expects (features x time), transpose each trial
        Y_dpad = [y.T for y in Y]
        Z_dpad = [z.T for z in Z] if Z is not None else None

        self.idSys = DPADModel()
        args = DPADModel.prepare_args(method_code)
        self.idSys.fit(Y_dpad, Z=Z_dpad, nx=nx, n1=n1, epochs=epochs, **args)
        return self.idSys

    def predict(self, Y: TrialList):
        """Predict on list of trials, handling each separately."""
        all_Zp, all_Yp, all_Xp = [], [], []

        for y_trial in Y:
            Zp, Yp, Xp = self.idSys.predict(y_trial)
            all_Zp.append(np.asarray(Zp) if Zp is not None else None)
            all_Yp.append(np.asarray(Yp) if Yp is not None else None)
            all_Xp.append(np.asarray(Xp) if Xp is not None else None)

        return all_Zp, all_Yp, all_Xp

    def validate(self, Y: TrialList) -> Dict[str, Any]:
        Zp, Yp, Xp = self.predict(Y)
        r_list, r_mean = pearson_r_per_channel(Y, Yp)
        result = {
            "Y": Y,
            "Zp": Zp,
            "Yp": Yp,
            "Xp": Xp,
            "Yp_shape": state_shape(Yp),
            "Zp_shape": (None if Zp is None else state_shape(Zp)),
            "Xp_shape": (None if Xp is None else state_shape(Xp)),
            "pearson_r_per_channel": r_list,
            "pearson_r_mean": r_mean,
        }
        return result

    def test(self, Y: TrialList) -> Dict[str, Any]:
        return self.validate(Y)

    def forecast(self, m: int, Y_past: Array2D):
        """Generate m-step ahead forecast using DPAD simulation mode.

        This does not require training with multi-step horizons. It sets the
        desired horizons [1..m], enables forward data-generation, calls predict,
        and extracts the last row from each horizon to form the m-step sequence.

        Args:
            m: number of steps to forecast
            Y_past: past observations (time x ny)

        Returns:
            Zf, Yf, Xf: arrays of shape (m, nz|ny|nx)
        """

        self.idSys.set_steps_ahead(list(range(1, m + 1)))
        self.idSys.set_multi_step_with_data_gen(True, noise_samples=0)
        preds = self.idSys.predict(Y_past)

        Z_steps = preds[:m]
        Y_steps = preds[m : 2 * m]
        X_steps = preds[2 * m : 3 * m]

        def _stack_last(steps_list):
            out = []
            for arr in steps_list:
                if arr is None:
                    out.append(None)
                    continue
                last_row = (
                    arr[-1:, :] if len(arr.shape) == 2 else np.atleast_2d(arr[-1])
                )
                out.append(last_row)
            if all(v is None for v in out):
                return None
            return np.vstack([v for v in out if v is not None])

        Yf = _stack_last(Y_steps)
        Xf = _stack_last(X_steps)
        Zf = _stack_last(Z_steps)

        return Zf, Yf, Xf

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate m-step ahead forecast.

        Note: Y_list and Z_list should already have margins removed (as done by _slice_data).
        The forecast window is the last m steps of each trial.

        Args:
            Y_list: list of trials (each T x ny ndarray), margins already removed.
            Z_list: optional list of behavioral trials (each T x nz ndarray), margins already removed.
            margin: Not used anymore since margins are pre-removed, kept for backward compatibility.
        """
        m = self.config.model.forecast.m
        margin_sec = margin if margin is not None else 0.0
        margin_samples = int(self.config.data.sampling_frequency * margin_sec)

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
        }

        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if m >= T:
                raise ValueError(
                    f"Forecast horizon m={m} must be smaller than trial length T={T}"
                )

            # Forecast the last m steps (margins already removed from Y_list)
            start = T - m
            end = T

            # Use all available past data (margins already removed)
            Y_past = Y[:start]
            Y_future_true = Y[start:end]

            # Get corresponding Z data if available
            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[start:end] if Z is not None else None

            Zf, Yf, Xf = self.forecast(m, Y_past)

            if Yf is not None:
                Y_concat = np.concatenate([Y_past, Yf], axis=0)
                r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
                r_list = r_list[0] if isinstance(r_list, list) else r_list
            else:
                Y_concat = Y_past
                r_list = []

            # Compute Z correlations if both true and predicted Z are available
            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = r_list_Z[0] if isinstance(r_list_Z, list) else r_list_Z
            else:
                r_list_Z = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist() if Yf is not None else None)
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(
                Z_future_true.tolist() if Z_future_true is not None else None
            )
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        # Compute overall mean for Y
        flat_r = [
            v
            for r in results["pearson_per_channel"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean"] = float(np.mean(flat_r)) if flat_r else np.nan

        # Compute overall mean for Z
        flat_r_Z = [
            v
            for r in results["pearson_per_channel_Z"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if flat_r_Z else np.nan
        )

        return results


class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)
