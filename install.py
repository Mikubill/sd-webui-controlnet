import launch
import pkg_resources
import sys
import os
import shutil
import platform
from pathlib import Path
from packaging import version
from typing import Tuple, Optional

# Define the root directory and main requirements file
repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"

def get_installed_version(package: str) -> Optional[str]:
    """Returns the installed version of a given package, if available."""
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None

def extract_base_package(package_string: str) -> str:
    """Extracts and returns the base package name from a package string."""
    return package_string.split("@git")[0].split("==")[0].split(">=")[0]

def install_requirements(req_file: Path):
    """Installs or upgrades packages based on a requirements.txt file."""
    with open(req_file) as file:
        for package in file:
            package = package.strip()
            package_name = extract_base_package(package)
            installed_version = get_installed_version(package_name)
            
            # If version is specified and not met, or package not installed, install/upgrade
            if installed_version is None or (package != package_name and not pkg_resources.require(package)):
                try:
                    launch.run_pip(f"install -U {package}", f"sd-webui-controlnet requirement: {package}")
                except Exception as e:
                    print(e)
                    print(f"Warning: Failed to install {package}, some preprocessors may not work.")

def try_install_from_wheel(pkg_name: str, wheel_url: str, version_req: Optional[str] = None):
    """Attempts to install or upgrade a package from a wheel URL."""
    current_version = get_installed_version(pkg_name)
    if version_req and (current_version is None or version.parse(current_version) < version.parse(version_req)):
        try:
            launch.run_pip(f"install -U {wheel_url}", f"sd-webui-controlnet requirement: {pkg_name}")
        except Exception as e:
            print(e)
            print(f"Warning: Failed to install {pkg_name}. Some processors may not work.")

def try_install_specific_packages():
    """Attempts to install specific packages with potential platform and version considerations."""
    # Example: Insightface installation
    if get_installed_version("insightface") is None:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        python_version = f"{sys.version_info.major}{sys.version_info.minor}"
        default_win_wheel = f"https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp{python_version}-cp{python_version}-win_{architecture}.whl"
        wheel_url = os.environ.get("INSIGHTFACE_WHEEL", default_win_wheel)
        try_install_from_wheel("insightface", wheel_url)

    # Add more specific package installations here as needed

def try_remove_legacy_submodule():
    """Attempts to remove a legacy submodule directory."""
    submodule = repo_root / "annotator" / "hand_refiner_portable"
    if submodule.exists():
        try:
            shutil.rmtree(submodule)
        except Exception as e:
            print(e)
            print(f"Failed to remove submodule {submodule} automatically. Please manually delete the directory.")

if __name__ == "__main__":
    install_requirements(main_req_file)
    try_install_specific_packages()
    try_remove_legacy_submodule()
