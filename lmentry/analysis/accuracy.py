import csv
import itertools
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lmentry.constants import (
    RESULTS_DIR, paper_models, get_short_model_name, PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR
)
from lmentry.tasks.lmentry_tasks import all_tasks, core_tasks

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def get_accuracy_and_certainty(task_name: str, model_name: str) -> dict:
    # load scored predictions
    prediction_path = PREDICTIONS_ROOT_DIR.joinpath(task_name).joinpath(f"{model_name}.json")
    if not prediction_path.exists():
        return dict()

    with open(prediction_path) as f_predictions:
        predictions = json.load(f_predictions)

    # load task data (for ids and templates)
    task_data_path = TASKS_DATA_DIR.joinpath(f"{task_name}.json")
    with open(task_data_path) as f_task:
        task_data = json.load(f_task)
    examples = task_data["examples"]
    settings = task_data["settings"]

    # get per-template counts of scores and certainty
    num_input_templates = len(settings["input_templates"])
    output = {f"template{i}": {"n_1s": 0,
                               "n_0s": 0,
                               "n_certain": 0,
                               "n_certain_1s": 0,
                               "n_certain_0s": 0,
                               }
              for i in range(num_input_templates)
              }
    for id_, prediction_entry in predictions.items():
        score = prediction_entry["score"]
        certainty = prediction_entry["certainty"]

        template_id = examples[id_]["metadata"]["template_id"]
        output[f"template{template_id}"][f"n_{score}s"] += 1
        output[f"template{template_id}"][f"n_certain"] += certainty
        if certainty:
            output[f"template{template_id}"][f"n_certain_{score}s"] += 1

    # calculate pre-template accuracy and certainty
    num_examples_per_template = task_data["settings"]["num_examples_per_template"]
    for template in output:
        output[template]["accuracy"] = output[template]["n_1s"] / num_examples_per_template
        output[template]["certainty"] = output[template]["n_certain"] / num_examples_per_template
        n_1s = output[template]["n_1s"]
        output[template]["certainty_of_1s"] = 0 if n_1s == 0 else output[template]["n_certain_1s"] / n_1s
        n_0s = output[template]["n_0s"]
        output[template]["certainty_of_0s"] = 0 if n_0s == 0 else output[template]["n_certain_0s"] / n_0s

        # round to one decimal digit
        for metric in ["accuracy", "certainty", "certainty_of_1s", "certainty_of_0s"]:
            output[template][metric] = round(output[template][metric] * 100, 1)

    task_results = dict()
    # calculate overall (across all templates) task counts of scores and certainty
    for metric in ["n_1s", "n_0s", "n_certain", "n_certain_1s", "n_certain_0s"]:
        task_results[metric] = sum([output[f"template{id_}"][metric]
                                    for id_ in range(num_input_templates)]
                                   )
    # calculate overall (across all templates) task accuracy and certainty
    n_task_examples = num_examples_per_template * num_input_templates
    task_results["accuracy"] = task_results["n_1s"] / n_task_examples
    task_results["certainty"] = task_results["n_certain"] / n_task_examples
    n_1s = task_results["n_1s"]
    task_results["certainty_of_1s"] = 0 if n_1s == 0 else task_results["n_certain_1s"] / n_1s
    n_0s = task_results["n_0s"]
    task_results["certainty_of_0s"] = 0 if n_0s == 0 else task_results["n_certain_0s"] / n_0s

    # round to two decimal digits
    for metric in ["accuracy", "certainty", "certainty_of_1s", "certainty_of_0s"]:
        task_results[metric] = round(task_results[metric] * 100, 2)

    output["task"] = task_results

    return output


def create_per_task_accuracy_csv(task_names: list[str] = None, model_names: list[str] = None,
                                 output_path: Path = None):
    rows: list[list] = list()

    model_names = model_names or list(paper_models)

    column_names = ["task"] + [get_short_model_name(model_name) for model_name in model_names]
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

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("accuracy_per_task.csv")
    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def create_per_template_accuracy_csv(task_names: list[str] = None, model_names: list[str] = None,
                                     output_path: Path = None):
    rows: list[list] = list()

    model_names = model_names or list(paper_models)

    first_row = ["task"]
    for model_name in model_names:
        first_row.extend([get_short_model_name(model_name)] * 3)  # replicating the model name for easy conversion to a multiindex df
    rows.append(first_row)

    # second row
    rows.append([""] + ["t0", "t1", "t2"] * len(model_names))

    # rest of the rows are task result rows
    task_names = task_names or list(all_tasks)
    for task_name in task_names:
        row = []
        row.append(task_name)

        # for each model, get the results for each template
        for model_name in model_names:
            metrics = get_accuracy_and_certainty(task_name, model_name)
            if not metrics:
                row.extend([""] * 3)
                logging.warning(f"no results for task {task_name} with model {model_name}")
                continue

            for template_name in ["template0", "template1", "template2"]:
                accuracy = metrics[template_name]["accuracy"]
                row.append(accuracy)

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("accuracy_per_template.csv")
    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def score_task_predictions(task_name: str, model_name: str):

    task = all_tasks[task_name]()
    task.score_predictions(model_name)


def score_all_predictions(task_names: list[str] = None, model_names: list[str] = None,
                          num_processes: int = 1):

    task_names = task_names or all_tasks.keys()
    model_names = model_names or list(paper_models)

    starargs = itertools.product(task_names, model_names)

    with Pool(processes=num_processes) as pool:
        pool.starmap(score_task_predictions, starargs)


def get_model_accuracy(model_name):

    task_accuracies = []
    for task_name in core_tasks:
        metrics = get_accuracy_and_certainty(task_name, model_name)
        if not metrics:
            raise ValueError(f"no results for {task_name} with model {model_name}, aborting accuracy calculation")
        else:
            task_accuracy = metrics["task"]["accuracy"]
            task_accuracies.append(task_accuracy)

    return np.mean(task_accuracies)
