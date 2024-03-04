import logging
import sys


def stream_logger_setup(params):
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(params['log']['log_level'])
    formatter = logging.Formatter(params['log']['log_format'])
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(params['log']['log_level'])

    return logger
