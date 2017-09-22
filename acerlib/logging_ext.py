# 1.0 - Acer 2017/06/09 15:28
import logging
import time


class DataLogger:
    def __init__(self, filename, dataFormat, header=None, mode='w'):
        """
        Create a data loggin obj
        :param filename: 
        :param dataFormat: a format string based on .format. e.g. ''{}, {}"
        :param header: a list of string 
        :param mode: 'w' or 'a'
        """
        self.filename = filename
        self.format = dataFormat
        self.header = header

        logger = logging.getLogger('DataLogger_{}'.format(time.clock()))

        # remove handler if exist
        if logger.hasHandlers():
            logger.handlers = []

        logger.setLevel(logging.DEBUG)

        # add new handler
        h = logging.FileHandler(filename, mode=mode)
        h.setFormatter(logging.Formatter('%(message)s'))
        h.setLevel(logging.DEBUG)
        logger.addHandler(h)
        logger.propagate = False

        if header is not None:
            logger.debug(dataFormat.format(*header))

        self.logger = logger

    def write(self, *args):
        s = self.format.format(*args)
        self.logger.debug(s)