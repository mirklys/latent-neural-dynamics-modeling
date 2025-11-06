from utils.config import Config
from typing import Any, Dict, List
import ..PSID.PSID as PSID
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel


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


class PSIDFramework(BaseFramework):
    def _initalize_model(self):
        self.logger.info("Initializing PSIDAdapter (function-based PSID API)")
        return PSIDWrapper(self.config)
