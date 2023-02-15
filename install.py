import launch

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirement for sd-webui-controlnet")
    
if not launch.is_installed("prettytable"):
    launch.run_pip("install prettytable", "requirement for sd-webui-controlnet")
