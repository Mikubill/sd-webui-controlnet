def preload(parser):
    # Setting default max_size=16 as each cache entry contains image as both key 
    # and value (Very costly).
    parser.add_argument(
        "--controlnet-preprocessor-cache-size",
        type=int,
        help="Cache size for controlnet preprocessor",
        default=16,
    )
