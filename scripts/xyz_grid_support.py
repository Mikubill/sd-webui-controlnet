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

    extra_axis_options = [
        xyz_grid.AxisOption("[ControlNet] Weight", float, apply_weight),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


xyz_grid = find_xyz_grid()

if xyz_grid:
    add_axis_options(xyz_grid)