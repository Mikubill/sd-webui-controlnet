import logging

from modules import shared

# Create a new logger
logger = logging.getLogger("ControlNet")

# Configure logger
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
loglevel_string = getattr(shared.cmd_opts, "controlnet_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), None)
logger.setLevel(loglevel)
