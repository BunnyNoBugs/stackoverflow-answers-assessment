import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config/config.yaml")


def load_config(config_path=CONFIG_PATH):
    """Load the configuration file."""
    with open(config_path) as file:
        config = yaml.safe_load(file)

    for d in config['data']:
        config['data'][d] = os.path.join(PROJECT_ROOT, config['data'][d])

    return config


CONFIG = load_config()
