from typing import List, Union

from tasks.lmentry.lmentry_tasks import (
  all_tasks as lmentry_all_tasks,
  core_tasks as lmentry_core_tasks,
  analysis_tasks as lmentry_analysis_tasks,
  sensetive_7b_model_tasks,
)
from tasks.simple.simple_tasks import simple_tasks
from tasks.leh.leh_tasks import leh_tasks


task_groups = {
  "lmentry": lmentry_all_tasks,
  "lmentry_core": lmentry_core_tasks,
  "lmentry_analysis": lmentry_analysis_tasks,
  "7b": sensetive_7b_model_tasks,
  "simple": simple_tasks,
  "leh": leh_tasks,
}


def search_task_obj_with_name(root_dict: dict, task_obj_name: str) -> Union[dict, str, None]:
  task_obj = None
  for name, elem in root_dict.items():
    # There are two option of element type: dict and Task class
    if task_obj_name == name:
      task_obj = root_dict[name]
      break
    elif isinstance(elem, dict):
      task_obj = search_task_obj_with_name(elem, task_obj_name)
      if task_obj: # not None
        break
  return task_obj


def get_all_task_names_from_dict(task_dict: dict) -> List[str]:
  ret_task_list = []
  for name, elem in task_dict.items():
    # There are two option of element type: dict and Task class
    if isinstance(elem, dict):
      ret_task_list = ret_task_list + get_all_task_names_from_dict(elem)
    else: # type(elem) == Task class
      ret_task_list.append(name)
  return sorted(ret_task_list)


def get_full_task_list():
  task_names = get_all_task_names_from_dict(task_groups)
  # remove duplicates
  task_names = list(set(task_names))
  return task_names


all_tasks = get_full_task_list()


# TODO(vvchernov): what task list is used by default?
def get_tasks_names(task_names: Union[str, List[str], None]=all_tasks):
  if not task_names:
    return None
  if isinstance(task_names, str):
    task_names = [task_names]
  ret_task_names = []
  for task_name in task_names:
    task_obj = search_task_obj_with_name(task_groups, task_name)
    if not task_obj: # task_obj == None
      raise ValueError(f"There is no task or task set with name {task_name}")
    elif isinstance(task_obj, dict):
      ret_task_names = ret_task_names + get_all_task_names_from_dict(task_obj)
    else:
      ret_task_names.append(task_name)
  return list(set(ret_task_names)) # remove duplicates


# TODO(vvchernov): it is LMentry mechanism, it is needed to update for other ones
def get_task(task_name: str):
  assert task_name in all_tasks, f"There is no task with name {task_name}"
  return search_task_obj_with_name(task_name)
