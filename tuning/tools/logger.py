# logger_config.py
import logging
import time
import os

DELIM = "=============================================="

# create dir if not exists
path = "/workspace/logs"
if not os.path.exists(path):
    print(f"Created dir {path}")
    os.makedirs(path)

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(f"{path}/training_{time.strftime('%Y%m%d%H%M%S')}.log"),
    logging.StreamHandler()
])

logging.info("Logging configuration is set up.")