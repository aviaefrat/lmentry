from pathlib import Path
from typing import List, Dict

from tasks.hf import hf_root_dir
from tasks.hf.prompts import load_prompt_list
from tasks.hf.utils import load_yaml_config
from tasks.hf.hf_task import HFTask
from tasks.hf.hf_task_config import HFTaskConfig


def check_prompt_config(config: Dict[str, str]) -> List[Dict[str, str]]:
  all_configs = []
  if "use_prompt" in config:
    prompt_list = load_prompt_list(
      use_prompt=config["use_prompt"],
      dataset_name=config["dataset_path"],
      subset_name=config["dataset_name"] if "dataset_name" in config else None,
    )
    for _, prompt_variation in enumerate(prompt_list):
      all_configs.append(
        {
          **config,
          **{"use_prompt": prompt_variation},
          **{
            "task": "_".join(
              [
                get_task_name_from_config(config),
                prompt_variation,
              ]
            )
          },
          **{"output_type": "greedy_until"},
        }
      )
  else:
    all_configs.append(config)
  return all_configs


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
  if "task" in task_config:
     return task_config["task"]
  elif "dataset_name" in task_config:
    return "{dataset_path}_{dataset_name}".format(**task_config)
  else:
    return "{dataset_path}".format(**task_config)


def get_tasks_dict(root_dir: Path):
  task_dict = {}
  for subdir in root_dir.iterdir():
    if subdir.is_dir() and subdir.name != "__pycache__":
      task_dict.update(get_tasks_dict(subdir))
  for yaml_path in root_dir.glob("*.yaml"):
    task_config = load_yaml_config(yaml_path)
    all_configs = check_prompt_config(task_config)
    for config in all_configs:
      # TODO(vvchernov): need extended config for child class by some metadata?
      task_name = get_task_name_from_config(config)
      local_dict = {task_name: type(f"{task_name}HFTask", (HFTask,), {"CONFIG": HFTaskConfig(**config)})}
      group_names = config.get("group", None)
      if not group_names:
        task_dict.update(local_dict)
      else:
        if isinstance(group_names, str):
          group_names = [group_names]
        for group_name in group_names:
          if group_name in task_dict.keys():
            task_dict[group_name].update(local_dict)
          else:
            task_dict[group_name] = local_dict
  return task_dict

hf_tasks = get_tasks_dict(hf_root_dir)
