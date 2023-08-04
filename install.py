import launch
import os
import pkg_resources
import subprocess

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    launch.run_pip(f"install {package}", f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"sd-webui-controlnet requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')


# DW Pose dependencies.
mim_packages = [
    "mmengine",
    "mmcv>=2.0.1",
    "mmdet>=3.1.0",
    "mmpose>=1.1.0",
]
for package in mim_packages:
    try:
        package_name = package.split('>=')[0]
        if not launch.is_installed(package_name):
            subprocess.call(["mim", "install", package])
    except Exception as e:
        print(f'Warning: Failed to install {package}. {e}')
