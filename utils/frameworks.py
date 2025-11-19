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
        self, Y_list: TrialList, margin: Optional[float] = None
    ) -> Dict[str, Any]:
        self.logger.info("Starting forecast validation...")
        return self.model.validate_forecast(Y_list, margin=margin)


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
        self, Y_list: TrialList, margin: Optional[float] = None
    ) -> Dict[str, Any]:
        """Validate m-step ahead forecast using config.model.forecast.m, with optional margin before the end.

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
        m = self.config.model.forecast.m
        margin_sec = margin if margin is not None else 0.0

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_pred": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
        }

        margin_samples = int(self.config.data.sampling_frequency * margin_sec)

        # Iterate trials
        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if m >= T:
                raise ValueError(
                    f"Forecast horizon m={m} must be smaller than trial length T={T}"
                )

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

            results["Y_future_true"].append(Y_future_true.tolist())
            results["margin_samples"] = margin_samples
            results["Y_future_pred"].append(Yf.tolist())
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
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
        self, Y_list: TrialList, margin: Optional[float] = None
    ) -> Dict[str, Any]:
        m = self.config.model.forecast.m
        margin_sec = margin if margin is not None else 0.0
        margin_samples = int(self.config.data.sampling_frequency * margin_sec)

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_pred": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
        }

        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if m >= T:
                raise ValueError(
                    f"Forecast horizon m={m} must be smaller than trial length T={T}"
                )

            # Compute indices: [-(m+margin_samples) : -margin_samples]
            start = T - (m + margin_samples)
            end = T - margin_samples

            if start < 0 or end <= start:
                raise ValueError(
                    f"Invalid margin for trial {idx}: T={T}, m={m}, margin={margin_samples}"
                )

            Y_past = Y[:start]
            Y_future_true = Y[start:end]
            Zf, Yf, Xf = self.forecast(m, Y_past)

            if Yf is not None:
                Y_concat = np.concatenate([Y_past, Yf], axis=0)
                r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
                r_list = r_list[0] if isinstance(r_list, list) else r_list
            else:
                Y_concat = Y_past
                r_list = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["margin_samples"] = margin_samples
            results["Y_future_pred"].append(Yf.tolist() if Yf is not None else None)
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)

        # Compute overall mean
        flat_r = [
            v
            for r in results["pearson_per_channel"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean"] = float(np.mean(flat_r)) if flat_r else np.nan
        return results


class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)
