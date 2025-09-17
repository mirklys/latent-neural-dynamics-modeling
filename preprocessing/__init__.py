from utils.package_configs import set_polars_config, logger_executor
from utils.config import get_config, Config


def initialize_preprocessing(config_path: str) -> Config:
    config_ = get_config(config_path)
    logger_executor(config_.logger_directory, name=config_.name)

    set_polars_config()

    return config_
