import launch
import os
import pkg_resources
from typing import Tuple, Optional

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")


def comparable_version(version: str) -> Tuple:
    return tuple(version.split('.'))
    

def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = get_installed_version(package_name)
                if installed_version != package_version:
                    launch.run_pip(f"install -U {package}", f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif '>=' in package:
                package_name, package_version = package.split('>=')
                installed_version = get_installed_version(package_name)
                if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                    launch.run_pip(f"install -U {package}", f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"sd-webui-controlnet requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')
