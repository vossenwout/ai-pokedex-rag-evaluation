import logging.config

import yaml


def init_logging(logging_config_path: str):
    with open(logging_config_path) as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
