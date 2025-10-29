from utils.config import Config
from typing import Any, Dict
import PSID
from utils.logger import get_logger


class BaseFramework:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = get_logger()

    def _initalize_model(self):
        raise NotImplemented

    def _train(self, Y, Z=None):
        y_shapes = [y.shape for y in Y] if isinstance(Y, (list, tuple)) else [Y.shape]
        z_info = (
            None
            if Z is None
            else ([z.shape for z in Z] if isinstance(Z, (list, tuple)) else [Z.shape])
        )
        self.logger.info(
            f"Initializing model and starting training. Y shapes (first 3): {y_shapes[:3]} | Z shapes (first 3): {z_info[:3] if z_info else None}"
        )
        self.model = self._initalize_model()
        self.logger.info(f"Model initialized: {self.model}")
        return self.model.train(Y, Z)

    def _validate(self, Y, Z=None):
        self.logger.info("Starting validation...")
        return self.model.validate(Y, Z)

    def _test(self, Y, Z=None):
        self.logger.info("Starting test...")
        return self.model.test(Y, Z)


class PSIDWrapper:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None
        self.params = _map_psid_params(config)

    def train(self, Y, Z=None):
        # Read configured parameters without additional shape-based adjustments
        nx = self.params["nx"]
        n1 = self.params.get("n1", nx)
        i = self.params["i"]
        time_first = self.params.get("time_first", True)
        remove_mean_Y = self.params.get("remove_mean_Y", True)
        remove_mean_Z = self.params.get("remove_mean_Z", True)
        zscore_Y = self.params.get("zscore_Y", False)
        zscore_Z = self.params.get("zscore_Z", False)

        y_shapes = [y.shape for y in Y] if isinstance(Y, (list, tuple)) else [Y.shape]
        z_shapes = (
            None
            if Z is None
            else ([z.shape for z in Z] if isinstance(Z, (list, tuple)) else [Z.shape])
        )
        self.logger.info(
            f"Calling PSID.PSID with nx={nx}, n1={n1}, i={i}, time_first={time_first}; "
            f"Y shapes (first 3): {y_shapes[:3]} Z shapes (first 3): {z_shapes[:3] if z_shapes else None}"
        )

        self.idSys = PSID.PSID(
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

    def validate(self, Y, Z=None):
        if self.idSys is None:
            raise RuntimeError("PSID model has not been trained yet.")
        if isinstance(Y, (list, tuple)) and len(Y) == 0:
            return {"status": "validation_skipped", "reason": "empty_validation_set"}

        Zp, Yp, Xp = self.idSys.predict(Y)
        result = {
            "Yp_shape": [p.shape for p in Yp] if isinstance(Yp, list) else Yp.shape,
            "Zp_shape": (
                None
                if Zp is None
                else (
                    [p.shape for p in Zp if p is not None]
                    if isinstance(Zp, list)
                    else Zp.shape
                )
            ),
            "Xp_shape": [p.shape for p in Xp] if isinstance(Xp, list) else Xp.shape,
        }

        return result

    def test(self, Y, Z=None):
        return self.validate(Y, Z)


class PSIDFramework(BaseFramework):
    def _initalize_model(self):
        self.logger.info("Initializing PSIDAdapter (function-based PSID API)")
        return PSIDWrapper(self.config)


def _map_psid_params(config: Config) -> Dict[str, Any]:
    model_cfg = config.model
    nx = int(model_cfg.nx)
    n1 = int(model_cfg.n1)
    i = int(model_cfg.i)

    time_first = bool(model_cfg.time_first)
    remove_mean_Y = bool(model_cfg.remove_mean_Y)
    remove_mean_Z = bool(model_cfg.remove_mean_Z)
    zscore_Y = bool(model_cfg.zscore_Y)
    zscore_Z = bool(model_cfg.zscore_Z)

    return {
        "nx": nx,
        "n1": n1,
        "i": i,
        "time_first": time_first,
        "remove_mean_Y": remove_mean_Y,
        "remove_mean_Z": remove_mean_Z,
        "zscore_Y": zscore_Y,
        "zscore_Z": zscore_Z,
    }


def _get_model_params(config: Config) -> Dict[str, Any]:
    return NotImplemented
