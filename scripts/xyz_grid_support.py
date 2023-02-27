import re

from modules import scripts, shared

try:
    from scripts import controlnet
except ImportError:
    import_error = True
else:
    import_error = False

DEBUG_MODE = False


def debug_info(func):
    def debug_info_(*args, **kwargs):
        if DEBUG_MODE:
            print(f"Debug info: {func.__name__}, {args}")
        return func(*args, **kwargs)
    return debug_info_


def find_xyz_grid():
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ["xyz_grid.py", "xy_grid.py"] and hasattr(data, "module"):
            return data.module

    return None


def add_axis_options(xyz_grid):
    # This class is currently meaningless.
    class AxisOption(xyz_grid.AxisOption):
        """This class returns an instance of the xyz_grid.AxisOption class.

        If clone is not specified, a single instance is returned.
        If clone is True, the number of copies is
        obtained from control_net_max_models_num and a list is returned.
        If clone is an integer greater than 1,
        that number of copies is made and returned as a list.
        """
        def __new__(cls, *args, clone=None, **kwargs):
            def init(this):
                cls.__init__(this, *args, **kwargs)
            if clone:
                this = super().__new__(cls)
                init(this)
                return this._clone(clone)
            else:
                this = super().__new__(xyz_grid.AxisOption)
                init(this)
                return this

        def _clone(self, num=True):
            def copy(self):
                return self.__class__(**self.__dict__)
            if num is True:
                num = shared.opts.data.get("control_net_max_models_num", 1)

            instance_list = [copy(self)]
            for i in range(1, num):
                instance = copy(self)
                instance.label = f"{self.label} - {i}"
                apply_func = self.apply.__kwdefaults__["enclosure"]
                apply_arg = f"{self.apply.__kwdefaults__['field']}_{i}"
                instance.apply = apply_func(apply_arg)
                instance_list.append(instance)

            return instance_list

    def normalize_list(valslist, type_func=None):
        """This function restores a broken list caused by the following process
        in the xyz_grid module.
            -> valslist = [x.strip() for x in chain.from_iterable(
                                                csv.reader(StringIO(vals)))]
        It also performs type conversion,
        adjusts the number of elements in the list, and other operations.
        """
        def search_bracket(string, bracket="[", replace=None):
            if bracket == "[":
                pattern = r"^\[(?![a-z0-9]{8}\])"
            elif bracket == "]":
                pattern = r"(?<!\[[a-z0-9]{8})\]$"
            else:
                raise ValueError(f"Invalid argument provided. (bracket: {bracket})")

            if replace is None:
                return re.search(pattern, string)
            else:
                return re.sub(pattern, replace, string)

        def type_convert(valslist, type_func, allow_blank=False):
            valslist_copy = valslist.copy()
            valslist.clear()
            for s in valslist_copy:
                if allow_blank and (s is None or s in ["None", ""]):
                    valslist.append(None)
                elif type_func:
                    valslist.append(type_func(s))
                else:
                    valslist.append(s)

        def fix(valslist, type_func):
            def is_same_length(list1, list2):
                return len(list1) == len(list2)

            fixing_list = []
            start_indices = []
            end_indices = []
            for i, s in enumerate(valslist):
                if is_same_length(start_indices, end_indices):
                    if search_bracket(s, "["):
                        s = search_bracket(s, "[", replace="")
                        start_indices.append(i)
                if not is_same_length(start_indices, end_indices):
                    if search_bracket(s, "]"):
                        s = search_bracket(s, "]", replace="")
                        end_indices.append(i + 1)
                fixing_list.append(s)

            if not is_same_length(start_indices, end_indices):
                raise ValueError(f"Lengths of {start_indices} and {end_indices} are different.")

            valslist[:] = fixing_list
            type_convert(valslist, type_func, allow_blank=True)

            # Restore the structure of a list.
            for i, j in zip(reversed(start_indices), reversed(end_indices)):
                valslist[i:j] = [valslist[i:j]]

        def padding(valslist):
            max_length = max(len(sub_list) for sub_list in valslist if isinstance(sub_list, list))
            for i, sub_list in enumerate(valslist):
                if isinstance(sub_list, list):
                    valslist[i] = sub_list + [None] * (max_length-len(sub_list))

        if not any(search_bracket(s) for s in valslist):  # There is no list inside
            type_convert(valslist, type_func, allow_blank=True)  # Type conv
            return
        else:                            # There is a list inside
            fix(valslist, type_func)  # Fix & Type conv
            padding(valslist)  # Fill sublist with None
            return

    ################################################
    ################################################
    #
    # Define a function to pass to the AxisOption class from here.
    #
    ################################################
    ################################################

    def enable_script_control():
        shared.opts.data["control_net_allow_script_control"] = True

    def apply_field(field):
        @debug_info
        def apply_field_(p, x, xs, *, field=field, enclosure=apply_field):
            enable_script_control()
            setattr(p, field, x)

        return apply_field_

    # Set this function as the type attribute of the AxisOption class.
    # To skip the following processing of xyz_grid module.
    #   -> valslist = [opt.type(x) for x in valslist]
    # Perform type conversion using the function
    # set to the confirm attribute instead.
    def identity(x):
        return x

    # The confirm function defined in this module
    # enables list notation and performs type conversion.
    #
    # Example:
    #     any = [any, any, any, ...]
    #     [any] = [any, None, None, ...]
    #     [None, None, any] = [None, None, any]
    #     [,,any] = [None, None, any]
    #     any, [,any,] = [any, any, any, ...], [None, any, None]
    #
    #     Enabled Only:
    #         any = [any] = [any, None, None, ...]
    #         (any and [any] are considered equivalent)
    def confirm(func_or_str):
        def find_dict(dict_name):
            return next((d for d in validation_data if d["name"] == dict_name), None)

        def flatten_list(lst):
            for element in lst:
                if isinstance(element, list):
                    yield from flatten_list(element)
                else:
                    yield element

        @debug_info
        def confirm_(p, xs):
            if callable(func_or_str):  # func_or_str is type_func
                normalize_list(xs, func_or_str)
                return

            elif isinstance(func_or_str, str):  # func_or_str is dict_name
                valid_data = find_dict(func_or_str)
                if valid_data is None:
                    return

                normalize_list(xs, valid_data["type"])

                confirm_list = valid_data["element"]()
                for x in flatten_list(xs):
                    if x is not None and x not in confirm_list:
                        raise RuntimeError(f"Unknown {valid_data['label']}: {x}")
                return

            else:
                return

        return confirm_

    def bool_(string):
        string = str(string)
        if string in ["None", ""]:
            return None
        elif string.lower() in ["true", "1"]:
            return True
        elif string.lower() in ["false", "0"]:
            return False
        else:
            raise ValueError(f"invalid literal for to_bool(): {string}")

    def choices_bool():
        return ["False", "True"]

    def choices_model():
        controlnet.update_cn_models()
        return list(controlnet.cn_models_names.values())

    def choices_preprocessor():
        return list(controlnet.Script().preprocessor)

    def choices_resize_mode():
        return ["Envelope (Outer Fit)", "Scale to Fit (Inner Fit)", "Just Resize"]

    validation_data = [
        {"name": "model", "type": str, "element": choices_model, "label": "ControlNet Model"},
        {"name": "resize_mode", "type": str, "element": choices_resize_mode, "label": "Resize Mode"},
        {"name": "preprocessor", "type": str, "element": choices_preprocessor, "label": "Preprocessor"},
    ]

    extra_axis_options = [
        AxisOption("[ControlNet] Enabled", identity, apply_field("control_net_enabled"), confirm=confirm(bool_), choices=choices_bool),
        AxisOption("[ControlNet] Model", identity, apply_field("control_net_model"), confirm=confirm("model"), choices=choices_model, cost=0.9),
        AxisOption("[ControlNet] Weight", identity, apply_field("control_net_weight"), confirm=confirm(float)),
        AxisOption("[ControlNet] Guidance Start", identity, apply_field("control_net_guidance_start"), confirm=confirm(float)),
        AxisOption("[ControlNet] Guidance End", identity, apply_field("control_net_guidance_end"), confirm=confirm(float)),
        AxisOption("[ControlNet] Resize Mode", identity, apply_field("control_net_resize_mode"), confirm=confirm("resize_mode"), choices=choices_resize_mode),
        AxisOption("[ControlNet] Preprocessor", identity, apply_field("control_net_module"), confirm=confirm("preprocessor"), choices=choices_preprocessor),
        AxisOption("[ControlNet] Pre Resolution", identity, apply_field("control_net_pres"), confirm=confirm(int)),
        AxisOption("[ControlNet] Pre Threshold A", identity, apply_field("control_net_pthr_a"), confirm=confirm(float)),
        AxisOption("[ControlNet] Pre Threshold B", identity, apply_field("control_net_pthr_b"), confirm=confirm(float)),
    ]

    xyz_grid.axis_options.extend(extra_axis_options)


def run():
    xyz_grid = find_xyz_grid()

    if xyz_grid:
        add_axis_options(xyz_grid)


if not import_error:
    run()
