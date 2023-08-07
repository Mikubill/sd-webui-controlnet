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


import subprocess
def has_nvidia_gpu() -> bool:
    """Returns whether Nvdia GPU is available on device by checking driver availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        if "NVIDIA" in result.stdout.decode():
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


def switch_onnxruntime(gpu_available: bool):
    """Switch between onnxruntime and onnxruntime-gpu based on GPU availability."""
    import pkg_resources
    def is_package_installed(package_name):
        try:
            pkg_resources.get_distribution(package_name)
            return True
        except pkg_resources.DistributionNotFound:
            return False
        
    if gpu_available:
        print("NVIDIA GPU detected. Using onnxruntime-gpu...")
        package = "onnxruntime-gpu"
        package_to_uninstall = "onnxruntime"
    else:
        print("No NVIDIA GPU detected. Using onnxruntime...")
        package = "onnxruntime"
        package_to_uninstall = "onnxruntime-gpu"
        
    if is_package_installed(package_to_uninstall):
        print(f"ControlNet: Uninstalling {package_to_uninstall}")
        subprocess.check_call(["pip", "uninstall", package_to_uninstall, "-y"])

    if not is_package_installed(package):
        print(f"ControlNet: Installing {package}")
        subprocess.check_call(["pip", "install", package])


# Guidelines on which package to use:
# - only install one of the two pip packages
# - the GPU package includes most CPU capabilities
# - stick to onnxruntime for ARM CPUs and/or macOS operating system
# https://github.com/microsoft/onnxruntime/issues/10685
switch_onnxruntime(has_nvidia_gpu())