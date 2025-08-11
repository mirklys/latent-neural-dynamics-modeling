import logging
import logging.config

import yaml

logging.config.dictConfig(
    yaml.load(open("./configs/logger_config.yaml", "r"), Loader=yaml.FullLoader)
)
logger = logging.getLogger("copydraw")
