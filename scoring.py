import argparse
import logging
from pathlib import Path

from lmentry.analysis.accuracy import score_all_predictions
from lmentry.constants import PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR
from lmentry.tasks.lmentry_tasks import all_tasks

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def parse_arguments():
  parser = argparse.ArgumentParser(
    description="CLI for scoring all or specified LMentry predictions using the default file locations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("-m", "--model_names", nargs="+", type=str, default=None,
                      help="Names of models which statistics were collected and should evaluate. "
                           "If it is None the predictions directory is analized and evailable model-statistics are evaluated")
  parser.add_argument('-t', '--task_names', nargs="+", type=str, default=None,
                      help="If need to evaluate specified set of tasks set their names. "
                           f"Task names should be from the list: {all_tasks.keys()}. "
                           "It tries to analyze all tasks by default")
  parser.add_argument("-n", "--num-procs", type=int, default=1,
                      help="The number of processes to use when scoring the predictions. "
                           "Can be up to the number of models you want to evaluate * 41.")
  return parser.parse_args()


def look_through_predictions_dir(model_names: list[str] = None, task_names: list[str] = None):
  model_tasks_dict = {}
  if task_names:
    tasks_path_list = [PREDICTIONS_ROOT_DIR.joinpath(task_name) for task_name in task_names]
    # Check
    checked_tasks_path_list = set(tasks_path_list)
    for task_path in tasks_path_list:
      if not task_path.is_dir():
        logging.warning(f"There is no {task_path.name} directory in predictions")
        checked_tasks_path_list.discard(task_path)
    tasks_path_list = list(checked_tasks_path_list)
  else:
    tasks_path_list = [f for f in PREDICTIONS_ROOT_DIR.iterdir() if f.is_dir()]
  for task_dir in tasks_path_list:
    task_name = task_dir.name
    if task_name in all_tasks:
      models_path_list =[]
      if model_names:
        models_path_list = [task_dir.joinpath(f"{model_name}.json") for model_name in model_names]
        # Check
        checked_models_path_list = set(models_path_list)
        for model_path in models_path_list:
          if not model_path.is_file():
            logging.warning(f"There is no {model_path.name} file in {task_name} directory")
            checked_models_path_list.discard(model_path)
        models_path_list = list(checked_models_path_list)
      else:
        models_path_list = [f for f in task_dir.iterdir() if f.is_file()]
      for model_path in models_path_list:
        if model_path.suffix == "json":
          model_name = model_path.stem
          if model_name in model_tasks_dict:
            model_tasks_dict[model_name].append(task_name)
          else:
            model_tasks_dict[model_name] = [task_name]
        else:
          logging.warning(f"Directory {task_name} in predictions is not a task!")
    else:
      logging.warning(f"Directory {task_name} in predictions is not a task!")
  return model_tasks_dict


def main():
  if not TASKS_DATA_DIR.exists():
      logging.error(f"LMentry tasks data not found at {TASKS_DATA_DIR}. aborting.\n")
      return
  if not PREDICTIONS_ROOT_DIR.exists():
      logging.error(f"Predictions not found at {PREDICTIONS_ROOT_DIR}. aborting.\n")
      return

  args = parse_arguments()

  model_tasks_dict = look_through_predictions_dir(model_names=args.model_names,
                                                  task_names=args.task_names)

  for model, tasks in model_tasks_dict:
    logging.info(f"scoring LMentry predictions for {model}")
    score_all_predictions(task_names=tasks,
                          model_names=[model],
                          num_processes=args.num_procs
                          )
    logging.info(f"Scoring LMentry predictions for {model} finished")


if __name__ == "__main__":
  main()
