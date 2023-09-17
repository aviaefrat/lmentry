import csv
import logging
from pathlib import Path
from tqdm import tqdm

from lmentry.tasks.lmentry_tasks import all_tasks
from lmentry.constants import RESULTS_DIR
from lmentry.analysis.accuracy import get_accuracy_and_certainty, get_comparison

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def create_per_task_accuracy_comparison_csv(
      model_names: list[str],
      task_names: list[str] = None,
      output_path: Path = None,
      certainty: bool=False):
  rows: list[list] = list()

  column_names = ["task"] + model_names + ["full match, %", "correct match, %", "wrong match, %", "correct non-match, %", "correct, %", "reduction, %"]
  column_num = len(column_names)
  rows.append(column_names)

  # rest of the rows are task result rows
  task_names = task_names or list(all_tasks)
  for task_name in tqdm(task_names, desc="Compare model accuracy for specified tasks"):
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
        if certainty:
          accuracy = metrics["task"]["certain_accuracy"]
        else:
          accuracy = metrics["task"]["accuracy"]
        row.append(accuracy)

    metrics = get_comparison(task_name, model_names, certainty)
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
  postfix=""
  if certainty:
    postfix = "_crtn"
  default_output_path = default_output_dir.joinpath(f"{model_names[0]}_VS_{model_names[1]}{postfix}.csv")
  if not output_path:
    default_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = default_output_path

  with open(output_path, "w", newline="") as f_output:
      writer = csv.writer(f_output)
      writer.writerows(rows)
