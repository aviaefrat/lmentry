import csv
import logging
from pathlib import Path

from lmentry.tasks.lmentry_tasks import all_tasks
from lmentry.constants import RESULTS_DIR
from lmentry.analysis.accuracy import get_accuracy_and_certainty, get_comparison
from lmentry.model_manager import get_type_config

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def create_per_task_accuracy_comparison_csv(
      model_names: list[str],
      task_names: list[str] = None,
      output_path: Path = None):
  rows: list[list] = list()

  short_model_names =[]
  for model_name in model_names:
    _, model_config = get_type_config(model_name)
    short_model_names.append(model_config["short_name"])
  model_names = short_model_names

  column_names = ["task"] + short_model_names + ["full match, %", "correct match, %", "wrong match, %", "correct non-match, %", "correct, %", "reduction, %"]
  column_num = len(column_names)
  rows.append(column_names)

  # rest of the rows are task result rows
  task_names = task_names or list(all_tasks)
  for task_name in task_names:
    row = []
    row.append(task_name)

    # for each model, get the results for each task
    for model_name in model_names:
      metrics = get_accuracy_and_certainty(task_name, model_name)
      if not metrics:
        row.append("")
        logging.warning(f"no results for task {task_name} with model {model_name}")
        continue
      else:
        accuracy = metrics["task"]["accuracy"]
        row.append(accuracy)
    
    metrics = get_comparison(task_name, model_names)
    reduction = "INF"
    if row[-2] > 0:
      reduction = round(100*row[-1]/row[-2], 2)
    row = row + [metrics["full match"], metrics["correct match"], metrics["wrong match"], metrics["correct non-match"], metrics["correct"], reduction]

    rows.append(row)

  # Average results:
  row_num = len(rows)
  avg_row = ["AVG"]
  for i_col in range(1, column_num):
    avg = 0
    for i_row in range(1, row_num):
      avg += rows[i_row][i_col]
    avg = round(avg/(row_num-1), 2)
    avg_row.append(avg)
  rows.append(avg_row)

  default_output_dir = RESULTS_DIR.joinpath("comparison/")
  default_output_path = default_output_dir.joinpath(f"{model_names[0]}_VS_{model_names[1]}.csv")
  if not output_path:
    default_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = default_output_path

  with open(output_path, "w", newline="") as f_output:
      writer = csv.writer(f_output)
      writer.writerows(rows)