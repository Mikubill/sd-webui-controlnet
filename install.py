import launch

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirement for sd-webui-controlnet")

if not launch.is_installed("svglib"):
    launch.run_pip("install svglib reportlab", "requirement for sd-webui-controlnet (svg_input)")
