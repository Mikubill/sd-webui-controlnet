def preload(parser):
    parser.add_argument("--controlnet-dir", type=str, help="Path to directory with ControlNet models", default=None)
    parser.add_argument("--controlnet-annotator-models-path", type=str, help="Path to directory with annotator model directories", default=None)
    parser.add_argument("--no-half-controlnet", action='store_true', help="do not switch the ControlNet models to 16-bit floats (only needed without --no-half)", default=None)
