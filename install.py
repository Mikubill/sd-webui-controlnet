import launch
import os

mim_packages = {
    "mmcv": "mmcv-full", 
    "mmseg": "mmsegmentation",
    "mmdet": "mmdet"
}
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

def run_mim(args, desc=None):
    if launch.skip_install:
        return
    
    if not launch.is_installed("mim"):
        launch.run_pip(f"install openmim", f"sd-webui-controlnet requirement: openmim")

    index_url_line = f' --index-url {launch.index_url}' if launch.index_url != '' else ''
    return launch.run(f'"{launch.python}" -m mim {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"sd-webui-controlnet requirement: {lib}")

# install mm packages
for pkg, lib in mim_packages.items():
    lib = lib.strip()
    if not launch.is_installed(pkg):
        run_mim(f"install {lib}", f"sd-webui-controlnet requirement: {lib}")