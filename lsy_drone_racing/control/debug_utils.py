import logging, json, pathlib, time
from datetime import datetime

LOGDIR = pathlib.Path("debug_logs")
LOGDIR.mkdir(exist_ok=True)

def get_logger(name: str = "mpcc"):
    log = logging.getLogger(name)
    if log.handlers:        
        return log
    log.setLevel(logging.DEBUG)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(LOGDIR / f"{name}_{stamp}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    log.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(sh)
    return log
