from abc import ABC, abstractmethod
from typing import List, ClassVar, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class PreprocessorParameter:
    """
    Class representing a parameter for a preprocessor.

    Attributes:
        label (str): The label for the parameter.
        minimum (float): The minimum value of the parameter. Default is 0.0.
        maximum (float): The maximum value of the parameter. Default is 1.0.
        step (float): The step size for the parameter. Default is 0.01.
        value (float): The initial value of the parameter. Default is 0.5.
        visible (bool): Whether the parameter is visible or not. Default is False.
    """

    label: str = "EMPTY_LABEL"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    value: float = 0.5
    visible: bool = True

    @property
    def gradio_update_kwargs(self) -> dict:
        return dict(
            minimum=self.minimum,
            maximum=self.maximum,
            step=self.step,
            label=self.label,
            value=self.value,
            visible=self.visible,
        )

    @property
    def api_json(self) -> dict:
        return dict(
            name=self.label,
            value=self.value,
            min=self.minimum,
            max=self.maximum,
            step=self.step,
        )


@dataclass
class Preprocessor(ABC):
    """
    Class representing a preprocessor.

    Attributes:
        name (str): The name of the preprocessor.
        tags (List[str]): The tags associated with the preprocessor.
        model_filename_filters (List[str]): The model filename filters for the preprocessor.
        slider_resolution (PreprocessorParameter): The parameter representing the resolution of the slider.
        slider_1 (PreprocessorParameter): The first parameter of the slider.
        slider_2 (PreprocessorParameter): The second parameter of the slider.
        slider_3 (PreprocessorParameter): The third parameter of the slider.
        show_control_mode (bool): Whether to show the control mode or not.
        do_not_need_model (bool): Whether the preprocessor needs a model or not.
        sorting_priority (int): The sorting priority of the preprocessor.
        corp_image_with_a1111_mask_when_in_img2img_inpaint_tab (bool): Whether to crop the image with a1111 mask when in img2img inpaint tab or not.
        fill_mask_with_one_when_resize_and_fill (bool): Whether to fill the mask with one when resizing and filling or not.
        use_soft_projection_in_hr_fix (bool): Whether to use soft projection in hr fix or not.
        expand_mask_when_resize_and_fill (bool): Whether to expand the mask when resizing and filling or not.
    """

    name: str
    _label: str = None
    tags: List[str] = field(default_factory=list)
    model_filename_filters: List[str] = field(default_factory=list)
    slider_resolution = PreprocessorParameter(
        label="Resolution",
        minimum=128,
        maximum=2048,
        value=512,
        step=8,
        visible=True,
    )
    slider_1 = PreprocessorParameter(visible=False)
    slider_2 = PreprocessorParameter(visible=False)
    slider_3 = PreprocessorParameter(visible=False)
    show_control_mode = True
    do_not_need_model = False
    sorting_priority = 0  # higher goes to top in the list
    corp_image_with_a1111_mask_when_in_img2img_inpaint_tab = True
    fill_mask_with_one_when_resize_and_fill = False
    use_soft_projection_in_hr_fix = False
    expand_mask_when_resize_and_fill = False

    all_processors: ClassVar[Dict[str, "Preprocessor"]] = {}
    all_processors_by_name: ClassVar[Dict[str, "Preprocessor"]] = {}

    @property
    def label(self) -> str:
        """Display name on UI."""
        return self._label if self._label is not None else self.name

    @classmethod
    def add_supported_preprocessor(cls, p: "Preprocessor"):
        assert p.label not in cls.all_processors, f"{p.label} already registered!"
        cls.all_processors[p.label] = p
        assert p.name not in cls.all_processors_by_name, f"{p.name} already registered!"
        cls.all_processors_by_name[p.name] = p

    @classmethod
    def get_preprocessor(cls, name: str) -> Optional["Preprocessor"]:
        return cls.all_processors.get(name, cls.all_processors_by_name.get(name, None))

    @classmethod
    def get_sorted_preprocessors(cls) -> List["Preprocessor"]:
        preprocessors = [p for k, p in cls.all_processors.items() if k != "none"]
        return [cls.all_processors["none"]] + sorted(
            preprocessors,
            key=lambda x: str(x.sorting_priority).zfill(8) + x.label,
            reverse=True,
        )

    @classmethod
    def get_all_preprocessor_tags(cls):
        tags = set()
        for _, p in cls.all_processors.items():
            tags.update(set(p.tags))
        return ["All"] + sorted(list(tags))

    @classmethod
    def get_filtered_preprocessors(cls, tag: str) -> List["Preprocessor"]:
        if tag == "All":
            return cls.all_processors
        return [
            p
            for p in cls.get_sorted_preprocessors()
            if tag in p.tags or p.label == "none"
        ]

    @classmethod
    def get_default_preprocessor(cls, tag: str) -> "Preprocessor":
        ps = cls.get_filtered_preprocessors(tag)
        assert len(ps) > 0
        return ps[0] if len(ps) == 1 else ps[1]

    @classmethod
    def tag_to_filters(cls, tag: str) -> Set[str]:
        return {
            f
            for p in cls.all_processors.values()
            if tag in p.tags
            for f in p.model_filename_filters
        }

    @abstractmethod
    def __call__(
        self,
        input_image,
        resolution,
        slider_1=None,
        slider_2=None,
        slider_3=None,
        input_mask=None,
        **kwargs,
    ):
        pass
