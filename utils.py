import logging
import os

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(save_dir, 'info.log'))
    fh.setLevel(logging.INFO)

    logger.addHandler(fh)

    return logger

