import launch

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirement for sd-webui-controlnet")
