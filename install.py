import launch
import subprocess
import pkg_resources
from pathlib import Path
from typing import Tuple, Optional


repo_root = Path(__file__).parent
main_req_file = repo_root / "requirements.txt"
hand_refiner_req_file = (
    repo_root / "annotator" / "hand_refiner_portable" / "requirements.txt"
)


def sync_submodules():
    try:
        subprocess.run(["git", "submodule", "init"], check=True, cwd=repo_root)
        subprocess.run(["git", "submodule", "update"], check=True, cwd=repo_root)
    except Exception as e:
        print(e)
        print("Warning: ControlNet failed to sync submodules. Please try run "
              "`git submodule init` and `git submodule update` manually.")


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


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
                elif not launch.is_installed(package):
                    launch.run_pip(
                        f"install {package}",
                        f"sd-webui-controlnet requirement: {package}",
                    )
            except Exception as e:
                print(e)
                print(
                    f"Warning: Failed to install {package}, some preprocessors may not work."
                )


sync_submodules()
install_requirements(main_req_file)
install_requirements(hand_refiner_req_file)
