# -*- coding: utf-8 -*-

import logging


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                                                   'message)s', datefmt='%H:%M:%S')
    return logging.getLogger(name)
