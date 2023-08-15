import csv
import itertools
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lmentry.constants import (
    RESULTS_DIR, paper_models, hf_models, PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR
)
from lmentry.tasks.lmentry_tasks import all_tasks, core_tasks

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def get_accuracy_and_certainty(task_name: str, model_name: str) -> dict:
    # load scored predictions
    prediction_path = PREDICTIONS_ROOT_DIR.joinpath(task_name).joinpath(f"{model_name}.json")
    if not prediction_path.exists():
        logging.warning(f"File {prediction_path} was not found therefore no results for task {task_name} with model {model_name}")
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


def get_short_model_names(model_names):
    short_model_names = []
    for model_name in model_names:
        short_name = ""
        if model_name in paper_models.keys():
            short_name = paper_models[model_name].get("short_name", model_name)
        elif model_name in hf_models.keys():
            short_name = hf_models[model_name].get("short_name", model_name)
        elif Path(model_name).is_dir():
            model_root = Path(model_name)
            mlc_config_file = model_root.joinpath("params/mlc-chat-config.json")

            mlc_config = {}
            with open(mlc_config_file) as json_file:
                mlc_config = json.load(json_file)

            model_local_id = mlc_config["local_id"]
            short_name = model_local_id.replace(".", "-")
        else:
            raise ValueError(f"Model name {model_name} is not in the list and not the path to mlc-llm model")

        short_model_names.append(short_name)
    return short_model_names


def create_per_task_accuracy_csv(task_names: list[str] = None, model_names: list[str] = None,
                                 output_path: Path = None):
    rows: list[list] = list()

    model_names = model_names or list(paper_models)

    column_names = ["task"] + get_short_model_names(model_names)
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
        first_row.extend(get_short_model_names([model_name]) * 3)  # replicating the model name for easy conversion to a multiindex df
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


def flexible_scoring(task_names: list[str] = None, model_names: list[str] = None,
                     num_processes: int = 1):
    if not TASKS_DATA_DIR.exists():
        logging.error(f"LMentry tasks data not found at {TASKS_DATA_DIR}. aborting.\n")
        return
    if not PREDICTIONS_ROOT_DIR.exists():
        logging.error(f"Predictions not found at {PREDICTIONS_ROOT_DIR}. aborting.\n")
        return

    model_tasks_dict = look_through_predictions_dir(model_names=model_names,
                                                    task_names=task_names)

    for model, tasks in model_tasks_dict.items():
        logging.info(f"scoring LMentry predictions for {model}")
        score_all_predictions(task_names=tasks,
                              model_names=[model],
                              num_processes=num_processes)
        logging.info(f"Scoring LMentry predictions for {model} finished")


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
