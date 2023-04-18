import os

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
os.makedirs(models_path, exist_ok=True)
