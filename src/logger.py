import logging


# Creates a logger with two handlers DEBUG->file, INFO->console
#   logger_name:    Logger name
#   filename:       Log file name. If None, no file is created.
def create_logger(logger_name, filename=None):
    # create logger with 'spam_application'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    if filename:
        fh = logging.FileHandler(filename, 'w')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger
