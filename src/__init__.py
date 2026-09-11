import os
import sys
import logging

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles default to a legacy codepage (cp1252) that can't encode the
    # emoji mlflow/dagshub write to stdout, crashing any script that imports this package.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("bmwmlopsLogger")
