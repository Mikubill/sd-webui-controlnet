import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        try:
            lib = lib.strip()
            if not launch.is_installed(lib):
                launch.run_pip(f"install {lib}", f"sd-webui-controlnet requirement: {lib}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {lib}, some preprocessors may not work.')
