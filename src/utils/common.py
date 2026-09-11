import json
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its contents as a ConfigBox (dot-accessible dict)."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file is empty: {path_to_yaml}")
    except Exception as e:
        logger.error(f"Failed to load yaml file {path_to_yaml}: {e}")
        raise


def create_directories(paths: list, verbose: bool = True) -> None:
    """Create each directory in `paths` if it doesn't already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict) -> None:
    """Save `data` as a JSON file at `path`."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")
