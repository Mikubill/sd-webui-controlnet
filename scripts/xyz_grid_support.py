from modules import scripts, shared
from scripts import controlnet


def find_xyz_grid():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ["xyz_grid.py", "xy_grid.py"] and hasattr(data, "module"):
            return data.module

    return None


def add_axis_options(xyz_grid):
    def enable_control_net(p):
        shared.opts.data["control_net_allow_script_control"] = True
        setattr(p, "control_net_enabled", True)

    def apply_field(field):
        def core(p, x, xs):
            enable_control_net(p)
            setattr(p, field, x)

        return core

    def choices_model():
        controlnet.update_cn_models()
        return list(controlnet.cn_models_names.values())

    def confirm_model(p, xs):
        confirm_list = choices_model()
        for x in xs:
            if x not in confirm_list:
                raise RuntimeError(f"Unknown ControlNet Model: {x}")

    def choices_resize_mode():
        return ["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"]

    def confirm_resize_mode(p, xs):
        confirm_list = choices_resize_mode()
        for x in xs:
            if x not in confirm_list:
                raise RuntimeError(f"Unknown Resize Mode: {x}")

    def choices_preprocessor():
        return list(controlnet.Script().preprocessor)

    def confirm_preprocessor(p, xs):
        confirm_list = choices_preprocessor()
        for x in xs:
            if x not in confirm_list:
                raise RuntimeError(f"Unknown Preprocessor: {x}")

    extra_axis_options = [
        xyz_grid.AxisOption("[ControlNet] Model", str, apply_field("control_net_model"), choices=choices_model, confirm=confirm_model, cost=0.9),
        xyz_grid.AxisOption("[ControlNet] Weight", float, apply_field("control_net_weight")),
        xyz_grid.AxisOption("[ControlNet] Guidance Strength", float, apply_field("control_net_guidance_strength")),
        xyz_grid.AxisOption("[ControlNet] Resize Mode", str, apply_field("control_net_resize_mode"), choices=choices_resize_mode),
        xyz_grid.AxisOption("[ControlNet] Preprocessor", str, apply_field("control_net_module"), choices=choices_preprocessor, confirm=confirm_preprocessor),
        xyz_grid.AxisOption("[ControlNet] Pre Resolution", int, apply_field("control_net_pres")),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold A", float, apply_field("control_net_pthr_a")),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold B", float, apply_field("control_net_pthr_b")),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


xyz_grid = find_xyz_grid()

if xyz_grid:
    add_axis_options(xyz_grid)
