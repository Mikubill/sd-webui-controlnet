import logging

from modules import shared

# Create a new logger
logger = logging.getLogger("ControlNet")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared.cmd_opts, "controlnet_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), None)
logger.setLevel(loglevel)
