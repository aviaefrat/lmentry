from pathlib import Path
from typing import List, Dict

from tasks.leh import leh_root_dir
from tasks.leh.prompts import load_prompt_list
from tasks.leh.utils import load_yaml_config


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
      # TODO(vvchernov): dummy None instead of Task class
      local_dict = {get_task_name_from_config(config): None}
      group_name = config.get("group", None)
      if not group_name:
        task_dict.update(local_dict)
      elif group_name in task_dict.keys():
        task_dict[group_name].update(local_dict)
      else:
        task_dict[group_name] = local_dict
  return task_dict

leh_tasks = get_tasks_dict(leh_root_dir)
