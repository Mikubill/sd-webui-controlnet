from modules import scripts, shared


def find_xyz_grid():
    for data in scripts.scripts_data:
        if data.script_class.__module__ == "xyz_grid.py":
            return data.module

    return None


def add_axis_options(xyz_grid):
    def enable_control_net(p):
        shared.opts.data["control_net_allow_script_control"] = True
        setattr(p, "control_net_enabled", True)

    def apply_weight(p, x, xs):
        enable_control_net(p)
        setattr(p, "control_net_weight", x)

    def apply_pres(p, x, xs):
        enable_control_net(p)
        setattr(p, "control_net_pres", x)

    def apply_pthr_a(p, x, xs):
        enable_control_net(p)
        setattr(p, "control_net_pthr_a", x)

    def apply_pthr_b(p, x, xs):
        enable_control_net(p)
        setattr(p, "control_net_pthr_b", x)

    def apply_resize_mode(p, x, xs):
        enable_control_net(p)
        setattr(p, "control_net_resize_mode", x)

    def choices_resize_mode():
        return ["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"]

    def confirm_resize_mode(p, xs):
        for x in xs:
            if x not in choices_resize_mode():
                raise RuntimeError(f"Unknown Resize Mode: {x}")

    extra_axis_options = [
        xyz_grid.AxisOption("[ControlNet] Weight", float, apply_weight),
        xyz_grid.AxisOption("[ControlNet] Pre Resolution", int, apply_pres),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold A", float, apply_pthr_a),
        xyz_grid.AxisOption("[ControlNet] Pre Threshold B", float, apply_pthr_b),
        xyz_grid.AxisOption("[ControlNet] Resize Mode", str, apply_resize_mode, confirm=confirm_resize_mode, choices=choices_resize_mode)
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


xyz_grid = find_xyz_grid()

if xyz_grid:
    add_axis_options(xyz_grid)
