from typing import List, Union

from tasks.lmentry.lmentry_tasks import (
  all_tasks as lmentry_all_tasks,
  core_tasks as lmentry_core_tasks,
  analysis_tasks as lmentry_analysis_tasks,
  sensetive_7b_model_tasks,
)
from tasks.simple.simple_tasks import simple_tasks


task_groups = {
  "lmentry": lmentry_all_tasks,
  "lmentry_core": lmentry_core_tasks,
  "lmentry_analysis": lmentry_analysis_tasks,
  "simple": simple_tasks,
  "7b": sensetive_7b_model_tasks,
}


def get_full_task_list():
  task_names = []
  for task_group in task_groups.values():
    task_names.append(task_group.keys())
  # remove duplicates
  task_names = list(set(task_names))
  return task_names


all_tasks = get_full_task_list()


# TODO(vvchernov): what task list is used by default?
def get_tasks_names(task_names: Union[str, List[str], None]=get_full_task_list()):
  if not task_names:
    return None
  if isinstance(task_names, str):
    task_names = [task_names]
  ret_task_names = []
  for task_name in task_names:
    if task_name in task_groups.keys():
      ret_task_names = ret_task_names + sorted(task_groups[task_name].keys())
    elif task_name in lmentry_all_tasks.keys():
      ret_task_names.append()
    else:
      raise ValueError(f"There is no task or task set with name {task_name}")
  return ret_task_names


# TODO(vvchernov): it is LMentry mechanism, it is needed to update for other ones
def get_task(task_name: str):
  assert task_name in get_full_task_list()
  for task_group in task_groups.values():
    for cur_task_name, cur_task in task_group.items():
      if cur_task_name == task_name:
        return cur_task
