from dataclasses import dataclass, asdict
import warnings

from typing import Union
from collections.abc import Callable


# TODO(vvchernov): it was copied from lm-evaluation-harness to support it here, need refactor and unification
@dataclass
class HFTaskConfig(dict):
  # task naming/registry
  task: str = None
  group: Union[str, list] = None
  # HF dataset options.
  # which dataset to use,
  # and what splits for what purpose
  dataset_path: str = None
  dataset_name: str = None
  dataset_kwargs: dict = None
  training_split: str = None
  validation_split: str = None
  test_split: str = None
  fewshot_split: str = None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaling (?)
  # formatting / prompting options.
  # see docs/advanced_task_guide.md for more info
  process_docs: Callable = None
  doc_to_text: Union[Callable, str] = None
  doc_to_target: Union[Callable, str] = None
  doc_to_choice: Union[Callable, str, dict, list] = None
  gold_alias: Union[Callable, str] = None
  process_results: Union[Callable, str] = None
  use_prompt: str = None
  description: str = ""
  target_delimiter: str = " "
  fewshot_delimiter: str = "\n\n"
  fewshot_config: dict = None
  # runtime configuration options
  num_fewshot: int = 0
  # scoring options
  metric_list: list = None
  output_type: str = "greedy_until"
  generation_kwargs: dict = None
  repeats: int = 1
  filter_list: Union[str, list] = None
  should_decontaminate: bool = False
  doc_to_decontamination_query: str = None

  metadata: str = None  # by default, not used in the code. allows for users to pass arbitrary info to tasks

  def __post_init__(self) -> None:
    if "." in self.dataset_path:
      import inspect
      from importlib import import_module

      self.dataset_path = inspect.getfile(import_module(self.dataset_path))

    if self.generation_kwargs is not None:
      if self.output_type != "greedy_until":
        warnings.warn(
          "passed `generation_kwargs`, but not using `output_type: greedy_until`!"
        )
        assert self.output_type != "greedy_until"

      if "temperature" in self.generation_kwargs:
        self.generation_kwargs["temperature"] = float(
          self.generation_kwargs["temperature"]
        )

      if "until" not in self.generation_kwargs:
        self.generation_kwargs["until"] = [self.fewshot_delimiter]
    else:
      if self.output_type == "greedy_until":
        # ensure that we greedily generate in absence of explicit arguments otherwise
        self.generation_kwargs = {
          "until": None
          if self.fewshot_delimiter is None
          else [self.fewshot_delimiter],
          "do_sample": False,
          "temperature": 0.0,
        }

  def __getitem__(self, item):
    return getattr(self, item)

  def __setitem__(self, item, value):
    return setattr(self, item, value)

  def to_dict(self):
    """dumps the current config as a dictionary object, as a printable format.
    null fields will not be printed.
    Used for dumping results alongside full task configuration

    :return: dict
        A printable dictionary version of the TaskConfig object.

    # TODO: should any default value in the TaskConfig not be printed?
    """
    cfg_dict = asdict(self)
    # remove values that are `None`
    for k, v in list(cfg_dict.items()):
      if v is None:
        cfg_dict.pop(k)
      elif isinstance(v, Callable):
        # TODO: this should handle Promptsource template objects as a separate case?
        cfg_dict[k] = str(v)
    return cfg_dict
