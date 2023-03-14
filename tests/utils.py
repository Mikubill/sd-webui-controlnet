import sys
import os

def remove_torch_check():
    try:
        skip_torch_index = sys.argv.index('--skip-torch-cuda-test')
        sys.argv[skip_torch_index:skip_torch_index + 1] = []
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    except ValueError:
        pass
