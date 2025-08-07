import logging

# Package-level logger
logger = logging.getLogger("ymcirc")
logger.debug("ymcirc logger initialized.")
logger.propagate = False  # To avoid duplication of log messages in root logger.
