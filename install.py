import launch
import pkg_resources
import sys
import os
import shutil
import platform
import requests
import tempfile
from pathlib import Path
from typing import Tuple, Optional


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
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


def try_install_hand_refiner():
    """ Attempt to install handrefinerportable. This library is necessary for depth_hand_refiner
    preprocessor. """
    if get_installed_version("handrefinerportable") is not None:
        return

    wheel_url =  os.environ.get("HANDREFINER_WHEEL", "https://github.com/huchenlei/HandRefinerPortable/releases/download/v1.0.0/handrefinerportable-2024.1.18.0-py2.py3-none-any.whl")
    try:
        launch.run_pip(
            f"install {wheel_url}",
            "sd-webui-controlnet requirement: handrefinerportable",
        )
    except Exception as e:
        print(e)
        print(
            "Warning: Failed to install handrefinerportable. depth_hand_refiner processor will not work."
        )


def try_install_insight_face():
    """Attempt to install insightface library. The library is necessary to use ip-adapter faceid.
    Note: Building insightface library from source requires compiling C++ code, which should be avoided
    in principle. Here the solution is to download a precompiled wheel."""
    if get_installed_version("insightface") is not None:
        return

    default_win_wheel = "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl"
    wheel_url = os.environ.get("INSIGHTFACE_WHEEL", default_win_wheel)

    system = platform.system().lower()
    architecture = platform.machine().lower()
    python_version = sys.version_info
    if wheel_url != default_win_wheel or (
        system == "windows"
        and "amd64" in architecture
        and python_version.major == 3
        and python_version.minor == 10
    ):
        try:
            launch.run_pip(
                f"install {wheel_url}",
                "sd-webui-controlnet requirement: insightface",
            )
        except Exception as e:
            print(
                "ControlNet init warning: Unable to install insightface automatically. "
                + e
            )
    else:
        print(
            "ControlNet init warning: Unable to install insightface automatically. "
            "Please try run `pip install insightface` manually."
        )


def try_remove_legacy_submodule():
    """Try remove annotators/hand_refiner_portable submodule dir."""
    submodule = repo_root / "annotator" / "hand_refiner_portable"
    if os.path.exists(submodule):
        try:
            shutil.rmtree(submodule)
        except Exception as e:
            print(e)
            print(
                f"Failed to remove submodule {submodule} automatically. You can manually delete the directory."
            )


install_requirements(main_req_file)
try_install_insight_face()
try_install_hand_refiner()
try_remove_legacy_submodule()
