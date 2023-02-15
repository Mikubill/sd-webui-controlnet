import os.path

EXTENSION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CN_DIR = os.path.join(EXTENSION_DIR, "models")

def preload(parser):
    parser.add_argument("--controlnet-dir", type=str, help="Path to directory with ControlNet models", default=DEFAULT_CN_DIR)
