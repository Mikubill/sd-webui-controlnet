from modules import scripts, shared
from scripts import controlnet


def find_xyz_grid():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ["xyz_grid.py", "xy_grid.py"] and hasattr(data, "module"):
            return data.module

    return None


def add_axis_options(xyz_grid):
    class AxisOption(xyz_grid.AxisOption):
        """This class returns an instance of the xyz_grid.AxisOption class.

        If clone is not specified, a single instance is returned.
        If clone is True, the number of copies is
        obtained from control_net_max_models_num and a list is returned.
        If clone is an integer greater than 1,
        that number of copies is made and returned as a list.
        """
        def __new__(cls, *args, clone=None, **kwargs):
            def init():
                # Compatible with older WebUI
                try:
                    cls.__init__(this, *args, **kwargs)
                except:
                    kwargs.pop("choices", None)
                    cls.__init__(this, *args, **kwargs)

            if clone:
                this = super().__new__(cls)
                init()
                return this._clone(clone)
            else:
                this = super().__new__(xyz_grid.AxisOption)
                init()
                return this

        def _clone(self, num=True):
            if num is True:
                num = shared.opts.data.get("control_net_max_models_num", 1)

            def copy():
                return self.__class__(**self.__dict__)

            instance_list = [copy()]

            for i in range(1, num):
                instance = copy()
                instance.label = f"{self.label} - {i}"

                apply_func = self.apply.__kwdefaults__["enclosure"]
                apply_arg = f"{self.apply.__kwdefaults__['field']}_{i}"
                instance.apply = apply_func(apply_arg)

                instance_list.append(instance)

            return instance_list

    def enable_control_net(p):
        shared.opts.data["control_net_allow_script_control"] = True
        setattr(p, "control_net_enabled", True)

    def apply_field(field):
        def core(p, x, xs, *, field=field, enclosure=apply_field):
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
        AxisOption("[ControlNet] Model", str, apply_field("control_net_model"), choices=choices_model, confirm=confirm_model, cost=0.9),
        AxisOption("[ControlNet] Weight", float, apply_field("control_net_weight")),
        AxisOption("[ControlNet] Guidance Start", float, apply_field("control_net_guidance_start")),
        AxisOption("[ControlNet] Guidance End", float, apply_field("control_net_guidance_end")),
        AxisOption("[ControlNet] Resize Mode", str, apply_field("control_net_resize_mode"), choices=choices_resize_mode, confirm=confirm_resize_mode),
        AxisOption("[ControlNet] Preprocessor", str, apply_field("control_net_module"), choices=choices_preprocessor, confirm=confirm_preprocessor),
        AxisOption("[ControlNet] Pre Resolution", int, apply_field("control_net_pres")),
        AxisOption("[ControlNet] Pre Threshold A", float, apply_field("control_net_pthr_a")),
        AxisOption("[ControlNet] Pre Threshold B", float, apply_field("control_net_pthr_b")),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


xyz_grid = find_xyz_grid()

if xyz_grid:
    add_axis_options(xyz_grid)
