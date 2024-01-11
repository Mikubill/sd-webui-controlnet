import launch
import git  # git is part of A1111 dependency.
import pkg_resources
import os
import sys
import platform
import requests
import tempfile
from pathlib import Path
from typing import Tuple, Optional


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"
hand_refiner_req_file = (
    repo_root / "annotator" / "hand_refiner_portable" / "requirements.txt"
)


def sync_submodules():
    try:
        repo = git.Repo(repo_root)
        repo.submodule_update()
    except Exception as e:
        print(e)
        print(
            "Warning: ControlNet failed to sync submodules. Please try run "
            "`git submodule init` and `git submodule update` manually."
        )


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    """trimesh[easy] -> trimesh"""
    # Split the string on '[' and take the first part
    base_package = package_string.split("[")[0]
    return base_package


def install_requirements(req_file):
    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if "==" in package:
                    package_name, package_version = package.split("==")
                    installed_version = get_installed_version(package_name)
                    if installed_version != package_version:
                        launch.run_pip(
                            f"install -U {package}",
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif ">=" in package:
                    package_name, package_version = package.split(">=")
                    installed_version = get_installed_version(package_name)
                    if not installed_version or comparable_version(
                        installed_version
                    ) < comparable_version(package_version):
                        launch.run_pip(
                            f"install -U {package}",
                            f"sd-webui-controlnet requirement: changing {package_name} version from {installed_version} to {package_version}",
                        )
                elif not launch.is_installed(extract_base_package(package)):
                    launch.run_pip(
                        f"install {package}",
                        f"sd-webui-controlnet requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some preprocessors may not work."
                )


def try_install_insight_face():
    """Attempt to install insightface library. The library is necessary to use ip-adapter faceid.
    Note: Building insightface library from source requires compiling C++ code, which should be avoided
    in principle. Here the solution is to download a precompiled wheel. """
    if get_installed_version("insightface") is not None:
        return

    def download_file(url, temp_dir):
        """ Download a file from a given URL to a temporary directory """
        local_filename = url.split('/')[-1]
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filepath = f"{temp_dir}/{local_filename}"
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        return filepath

    def install_wheel(wheel_path):
        """Install the wheel using pip"""
        launch.run_pip(
            f"install {wheel_path}",
            f"sd-webui-controlnet requirement: install insightface",
        )

    wheel_url = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"

    system = platform.system().lower()
    architecture = platform.machine().lower()
    python_version = sys.version_info
    if (
        system == "windows"
        and "amd64" in architecture
        and python_version.major == 3
        and python_version.minor == 10
    ):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(
                    "Downloading the prebuilt wheel for Windows amd64 to a temporary directory..."
                )
                wheel_path = download_file(wheel_url, temp_dir)
                print(f"Download complete. File saved to {wheel_path}")

                print("Installing the wheel...")
                install_wheel(wheel_path)
                print("Installation complete.")
        except Exception as e:
            print(
                "ControlNet init warning: Unable to install insightface automatically. " + e
            )
    else:
        print(
            "ControlNet init warning: Unable to install insightface automatically. "
            "Please try run `pip install insightface` manually."
        )


sync_submodules()
install_requirements(main_req_file)
if os.path.exists(hand_refiner_req_file):
    install_requirements(hand_refiner_req_file)
try_install_insight_face()
