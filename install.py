import launch
import os
import pkg_resources

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
def install_mim():
    import importlib
    module_spec = importlib.util.find_spec('mim')
    if module_spec is None:
        launch.run_pip("install openmim", "sd-webui-controlnet requirement: openmim")

mim_packages = [
    "mmengine",
    "mmcv>=2.0.1",
    "mmdet>=3.1.0",
    "mmpose>=1.1.0",
]

def install_mim_packages(packages):
    import mim
    for package in mim_packages:
        packages = {p for (p, *_) in mim.list_package()}
        try:
            package_name = package.split('>=')[0]
            if package_name not in packages:
                mim.install([package])
        except Exception as e:
            print(f'Warning: Failed to install {package}. {e}')
            raise e

install_mim()
install_mim_packages(mim_packages)