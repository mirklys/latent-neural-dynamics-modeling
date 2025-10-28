class BaseFramework:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _initalize_model(self):
        raise NotImplemented

    def _train(self, Y, Z=None, U=None):
        self.model = self._initalize_model()

        return self.model.train(Y, Z, U)

    def _validate(self, Y, Z=None, U=None):
        return self.model.validate(Y, Z, U)

    def _test(self, Y, Z=None, U=None):
        return self.model.test(Y, Z, U)


class PSIDFramework(BaseFramework):
    def _initalize_model(self):
        kwargs = _get_model_params(self.config)
        return PSID(**kwargs)


class DPADFramework(BaseFramework):
    def _initalize_model(self):
        kwargs = _get_model_params(self.config)
        return DPAD(**kwargs)


def _common_model_kwargs(config: Config) -> Dict[str, Any]:
    model_cfg = config.model
    kwargs = {
        "n_states": int(model_cfg.get("n_states", 20)),
        "past_horizon": model_cfg.get("past_horizon", 10),
        "future_horizon": model_cfg.get("future_horizon", 10),
        "rank_n": model_cfg.get("rank_n", None),
        "alpha": model_cfg.get("alpha", None),
        "stable_A": model_cfg.get("stable_A", False),
        "estimate_noise": model_cfg.get("estimate_noise", True),
    }
    return {k: v for k, v in kwargs.items() if v is not None}
