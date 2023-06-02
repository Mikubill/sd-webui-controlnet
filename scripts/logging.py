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


if hasattr(shared.cmd_opts, "controlnet_loglevel"):
    loglevel_string = shared.cmd_opts.controlnet_loglevel
else:
    print("Warning: `controlnet_loglevel` not available from A1111")
    loglevel_string = "DEBUG"

loglevel = getattr(logging, loglevel_string.upper(), None)
logger.setLevel(loglevel)
