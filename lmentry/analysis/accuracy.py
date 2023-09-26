import csv
import itertools
import json
import logging
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import numpy as np

from lmentry.constants import (
    RESULTS_DIR, paper_models, PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR, HF_TASKS_DATA_DIR
)
from tasks.lmentry.lmentry_tasks import core_tasks
from tasks.task_utils import all_tasks, all_hf_task, get_task

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def get_prediction(task_name: str, model_name: str):
    prediction_path = PREDICTIONS_ROOT_DIR.joinpath(task_name).joinpath(f"{model_name}.json")
    if not prediction_path.exists():
        logging.warning(f"File {prediction_path} was not found therefore no results for task {task_name} with model {model_name}")
        return dict()

    with open(prediction_path) as f_predictions:
        predictions = json.load(f_predictions)

    return predictions


def get_accuracy_and_certainty(task_name: str, model_name: str) -> dict:
    # load scored predictions
    predictions = get_prediction(task_name, model_name)

    # load task data (for ids and templates)
    # TODO(vvchernov): need more flexible approach
    if task_name in all_hf_task:
        task_data_path = HF_TASKS_DATA_DIR.joinpath(f"{task_name}.json")
    else:
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
        if id_ != "scoring":
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
    task_results["certain_accuracy"] = task_results["n_certain_1s"] / n_task_examples
    task_results["certainty"] = task_results["n_certain"] / n_task_examples
    n_1s = task_results["n_1s"]
    task_results["certainty_of_1s"] = 0 if n_1s == 0 else task_results["n_certain_1s"] / n_1s
    n_0s = task_results["n_0s"]
    task_results["certainty_of_0s"] = 0 if n_0s == 0 else task_results["n_certain_0s"] / n_0s

    # round to two decimal digits
    for metric in ["accuracy", "certain_accuracy", "certainty", "certainty_of_1s", "certainty_of_0s"]:
        task_results[metric] = round(task_results[metric] * 100, 2)

    output["task"] = task_results

    return output


def get_score(entry, certainty=False):
    if certainty and entry["certainty"] != 1:
        return 0
    return entry["score"]


def get_comparison(task_name: str, model_names: list[str], certainty=False) -> dict:
    metrics={
        "full match": 0,
        "correct match": 0,
        "wrong match": 0,
        "correct non-match": 0,
        "correct": 0,
    }

    if len(model_names) != 2:
        logging.error(f"Two model names for comparison are expeceted, but {len(model_names)} were given")
        return dict()
    ref_model_name = model_names[0]
    probe_model_name = model_names[1]

    # load scored predictions for reference model
    ref_predictions = get_prediction(task_name, ref_model_name)

    # load scored predictions for probe model
    probe_predictions = get_prediction(task_name, probe_model_name)

    full_counter = 0
    full_match_counter = 0
    correct_match_counter = 0
    wrong_match_counter = 0
    correct_non_match_counter = 0
    correct_counter = 0
    correct_ref_counter = 0
    for id_ in ref_predictions:
        if id_ != "scoring":
            full_counter += 1
            ref_prediction_entry = ref_predictions[id_]
            probe_prediction_entry = probe_predictions[id_]
            ref_prediction = ref_prediction_entry["prediction"].lower()
            probe_prediction = probe_prediction_entry["prediction"].lower()
            ref_score = get_score(ref_prediction_entry, certainty)
            probe_score = get_score(probe_prediction_entry, certainty)
            if ref_score == 1:
                correct_ref_counter += 1
            if ref_prediction == probe_prediction:
                full_match_counter += 1
                if ref_score == probe_score:
                    if ref_score == 1:
                        correct_match_counter += 1
                        correct_counter += 1
                    elif ref_score == 0:
                        wrong_match_counter += 1
            elif ref_score == 1 and probe_score == 1:
                correct_non_match_counter += 1
                correct_counter += 1

    if full_counter == 0:
        logging.warning("There is no any predictions")
        full_counter = 1
    if correct_ref_counter == 0:
        logging.warning("There is no any correct predictions")
        correct_ref_counter = 1
    wrong_ref_counter = full_counter - correct_ref_counter
    if wrong_ref_counter < 1:
        logging.warning("Number of wrong predictions is not positive")
        wrong_ref_counter = 1
    metrics["full match"] = float(full_match_counter) / full_counter
    metrics["correct match"] = float(correct_match_counter) / correct_ref_counter
    metrics["wrong match"] = float(wrong_match_counter) / wrong_ref_counter
    metrics["correct non-match"] = float(correct_non_match_counter) / correct_ref_counter
    metrics["correct"] = float(correct_counter) / correct_ref_counter

    # round to two decimal digits
    for metric_type in ["full match", "correct match", "wrong match", "correct non-match", "correct"]:
        metrics[metric_type] = round(metrics[metric_type] * 100, 2)

    return metrics


def create_per_task_accuracy_csv(task_names: list[str] = None, model_names: list[str] = None,
                                 output_path: Path = None):
    rows: list[list] = list()

    model_names = model_names or list(paper_models)

    column_names = ["task"] + model_names
    rows.append(column_names)

    # rest of the rows are task result rows
    task_names = task_names or all_tasks
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
                accuracy = metrics["task"]["accuracy"]
                row.append(accuracy)

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("accuracy_per_task.csv")
    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def create_per_template_accuracy_csv(task_names: list[str] = None, model_names: list[str] = None,
                                     output_path: Path = None, template_num: int = 3):
    rows: list[list] = list()

    model_names = model_names or list(paper_models)

    first_row = ["task"]
    for model_name in model_names:
        first_row.extend([model_name] * template_num)  # replicating the model name for easy conversion to a multiindex df
    rows.append(first_row)

    # second row
    template_tags = []
    for i in range(template_num):
        template_tags.append(f"t{i}")
    # TODO(vchernov): create more flexible tags, remove absence templates
    rows.append([""] + template_tags * len(model_names))

    # rest of the rows are task result rows
    task_names = task_names or all_tasks
    template_names = []
    for i in range(template_num):
        template_names.append(f"template{i}")
    for task_name in tqdm(task_names, desc="Compare model accuracy for specified tasks"):
        row = []
        row.append(task_name)

        # for each model, get the results for each template
        for model_name in model_names:
            metrics = get_accuracy_and_certainty(task_name, model_name)
            if not metrics:
                row.extend([""] * template_num)
                logging.warning(f"no results for task {task_name} with model {model_name}")
                continue

            for template_name in template_names:
                # Different tasks can have different number of templates
                # if there is no template zero is returned
                template_metrics = metrics.get(template_name, None)
                if template_metrics:
                    accuracy = template_metrics["accuracy"]
                else:
                    accuracy = 0.
                row.append(accuracy)

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("accuracy_per_template.csv")
    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def score_task_predictions(task_name: str, model_name: str, forced_scoring: bool=False):
    task = get_task(task_name)
    task.score_predictions(model_name, forced_scoring=forced_scoring)


def score_all_predictions(task_names: list[str] = None, model_names: list[str] = None,
                          num_processes: int = 1, forced_scoring: bool=False):

    task_names = task_names or all_tasks
    model_names = model_names or list(paper_models)
    starargs = itertools.product(task_names, model_names, [forced_scoring])

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
                if model_path.suffix == ".json":
                    model_name = model_path.stem
                    if model_name in model_tasks_dict:
                        model_tasks_dict[model_name].append(task_name)
                    else:
                        model_tasks_dict[model_name] = [task_name]
                else:
                    logging.warning(f"File {model_path} in {task_name} directory is not a json-file!")
        else:
            logging.warning(f"Directory {task_name} in predictions is not a task!")
    return model_tasks_dict


def flexible_scoring(task_names: list[str] = None, model_names: list[str] = None,
                     num_processes: int = 1, forced_scoring: bool=False):
    if not TASKS_DATA_DIR.exists():
        logging.error(f"LMentry tasks data not found at {TASKS_DATA_DIR}. aborting.\n")
        return
    if not PREDICTIONS_ROOT_DIR.exists():
        logging.error(f"Predictions not found at {PREDICTIONS_ROOT_DIR}. aborting.\n")
        return

    model_tasks_dict = look_through_predictions_dir(model_names=model_names,
                                                    task_names=task_names)

    for model, tasks in tqdm(model_tasks_dict.items(), desc="Score models"):
        logging.info(f"scoring LMentry predictions for {model}")
        score_all_predictions(task_names=tasks,
                              model_names=[model],
                              num_processes=num_processes,
                              forced_scoring=forced_scoring,)
        logging.info(f"Scoring LMentry predictions for {model} finished")


def get_model_accuracy(model_name):
    task_accuracies = []
    # TODO(vvchernov): in general case it is not neccessary to be lmentry_core_tasks only
    for task_name in core_tasks:
        metrics = get_accuracy_and_certainty(task_name, model_name)
        if not metrics:
            raise ValueError(f"no results for {task_name} with model {model_name}, aborting accuracy calculation")
        else:
            task_accuracy = metrics["task"]["accuracy"]
            task_accuracies.append(task_accuracy)

    return np.mean(task_accuracies)
