import csv
import json
import logging
from functools import partial
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd

from lmentry.analysis.accuracy import get_accuracy_and_certainty
from lmentry.constants import (
    simple_answer_index_task_names, RESULTS_DIR, paper_models, get_short_model_name,
    PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR
)
from tasks.lmentry.lmentry_tasks import core_tasks, all_tasks

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


argument_content_robustness_task_argument_subsets = {
    "first_alphabetically": [
        "far_first_letter",
        "different_first_letter",
        "consecutive_first_letter",
        "same_first_letter"
    ],
    "all_words_from_category": [
        "0_distractors",
        "1_distractors",
        "2_distractors"
    ],
    "any_words_from_category": [
        "5_distractors",
        "4_distractors",
        "3_distractors"
    ],
    "rhyming_word": [
        "orthographically_similar",
        "orthographically_different"
    ],
    "more_letters": [
        "length_diff_3plus",
        "length_diff_1"
    ],
    "less_letters": [
        "length_diff_3plus",
        "length_diff_1"
    ]
}

adjacent_tasks_by_definition_pairs = {
    ("bigger_number", "smaller_number"): "bigger/smaller_number",
    ("more_letters", "less_letters"): "more/less_letters",
    ("word_after", "word_before"): "word_after/before",
    ("most_associated_word", "least_associated_word"): "most/least_associated_word",
    ("any_words_from_category", "all_words_from_category"): "any/all_words_from_category"
}
adjacent_tasks_explicit_negation_pairs = {
    ("sentence_containing", "sentence_not_containing"): "sentence_(not)_containing",
    ("word_containing", "word_not_containing"): "word_(not)_containing"
}
adjacent_tasks_character_awareness_pairs = {
    ("starts_with_word", "starts_with_letter"): "starts_with_word/letter",
    ("ends_with_word", "ends_with_letter"): "ends_with_word/letter",
    ("sentence_containing", "word_containing"): "sentence/word_containing",
    ("sentence_not_containing", "word_not_containing"): "sentence/word_not_containing",
    ("first_word", "first_letter"): "first_word/letter",
    ("last_word", "last_letter"): "last_word/letter"
}
adjacent_tasks_by_first_vs_last_pairs = {
    ("starts_with_word", "ends_with_word"): "starts/ends_with_word",
    ("starts_with_letter", "ends_with_letter"): "starts/ends_with_letter",
    ("first_word", "last_word"): "first/last_word",
    ("first_letter", "last_letter"): "first/last_letter"
}
adjacent_tasks_tasks_pairs = {
    **adjacent_tasks_by_definition_pairs,
    **adjacent_tasks_explicit_negation_pairs,
    **adjacent_tasks_character_awareness_pairs,
    **adjacent_tasks_by_first_vs_last_pairs
}


def robustness_func(accs: list[float]):

    max_difference = 0
    for acc1, acc2 in combinations(accs, 2):
        diff = abs(acc1 - acc2)
        if diff > max_difference:
            max_difference = diff

    return 100 - max_difference


def get_simple_answer_index_bias(task_name: str, model_name: str) -> dict:

    # load scored predictions
    prediction_path = PREDICTIONS_ROOT_DIR.joinpath(task_name).joinpath(f"{model_name}.json")
    if not prediction_path.exists():
        logging.warning(f"No predictions exist in {prediction_path}")
        return dict()

    with open(prediction_path) as f_predictions:
        predictions = json.load(f_predictions)

    # load task data (for template_id and answer_index)
    task_data_path = TASKS_DATA_DIR.joinpath(f"{task_name}.json")
    with open(task_data_path) as f_task:
        task_data = json.load(f_task)
    examples = task_data["examples"]
    settings = task_data["settings"]

    # get per-template number of correct answers w.r.t answer index
    num_input_templates = len(settings["input_templates"])
    output = {f"template{i}": {"n_correct": 0,
                               "n_answer_index_0": 0,
                               "n_answer_index_1": 0,
                               "n_correct_answer_index_0": 0,
                               "n_correct_answer_index_1": 0,
                               }
              for i in range(num_input_templates)
              }
    for id_, prediction_entry in predictions.items():
        score = prediction_entry["score"]
        answer_index = examples[id_]["metadata"]["answer_index"]
        template_id = examples[id_]["metadata"]["template_id"]
        output[f"template{template_id}"][f"n_answer_index_{answer_index}"] += 1
        if score == 1:
            output[f"template{template_id}"][f"n_correct"] += 1
            output[f"template{template_id}"][f"n_correct_answer_index_{answer_index}"] += 1

    # calculate pre-template accuracy and accuracy w.r.t answer index
    num_examples_per_template = task_data["settings"]["num_examples_per_template"]
    for template in output:
        output[template]["accuracy"] = output[template]["n_correct"] / num_examples_per_template
        for answer_index in [0, 1]:
            n_answer_index = output[template][f"n_answer_index_{answer_index}"]
            n_correct_answer_index = output[template][f"n_correct_answer_index_{answer_index}"]
            output[template][f"accuracy_answer_index_{answer_index}"] = (
                0 if n_correct_answer_index == 0
                else n_correct_answer_index / n_answer_index
            )
        # round to one decimal digit
        for metric in ["accuracy", "accuracy_answer_index_0", "accuracy_answer_index_1"]:
            output[template][metric] = round(output[template][metric] * 100, 1)

    task_results = dict()
    # calculate overall (across all templates) counts
    for metric in ["n_correct", "n_answer_index_0", "n_answer_index_1",
                   "n_correct_answer_index_0", "n_correct_answer_index_1"]:
        task_results[metric] = sum([output[f"template{id_}"][metric]
                                    for id_ in range(num_input_templates)]
                                   )
    # calculate overall (across all templates) accuracy and accuracy w.r.t answer index
    n_task_examples = num_examples_per_template * num_input_templates
    task_results["accuracy"] = task_results["n_correct"] / n_task_examples
    for answer_index in [0, 1]:
        n_answer_index = task_results[f"n_answer_index_{answer_index}"]
        n_correct_answer_index = task_results[f"n_correct_answer_index_{answer_index}"]
        task_results[f"accuracy_answer_index_{answer_index}"] = (
            0 if n_correct_answer_index == 0
            else n_correct_answer_index / n_answer_index
        )
    # round to two decimal digits
    for metric in ["accuracy", "accuracy_answer_index_0", "accuracy_answer_index_1"]:
        task_results[metric] = round(task_results[metric] * 100, 2)

    output["task"] = task_results

    return output


def get_task_argument_order_robustness(task_name: str, model_name: str, func: Callable = robustness_func):

    answer_index_metrics = get_simple_answer_index_bias(task_name, model_name)
    accuracy_index_0 = answer_index_metrics["task"]["accuracy_answer_index_0"]
    accuracy_index_1 = answer_index_metrics["task"]["accuracy_answer_index_1"]

    robustness = func([accuracy_index_0, accuracy_index_1])

    return robustness


def get_argument_order_robustness(model_name: str, task_names: list[str] = None,
                                  func: Callable = robustness_func):

    task_names = task_names or simple_answer_index_task_names

    return np.mean([
        get_task_argument_order_robustness(task_name=task_name, model_name=model_name, func=func)
        for task_name in task_names
    ])


def get_argument_content_robustness(model_name: str, func: Callable = robustness_func,
                                    tasks_argument_subsets: dict[str, list[str]] = None):

    tasks_argument_subsets = tasks_argument_subsets or argument_content_robustness_task_argument_subsets

    argument_content_biases = []

    for task_name, argument_subset_names in tasks_argument_subsets.items():

        task_group_accuracies = []
        for argument_subset_name in argument_subset_names:
            metrics = get_accuracy_and_certainty(f"{task_name}_{argument_subset_name}", model_name)
            task_accuracy = metrics["task"]["accuracy"]
            task_group_accuracies.append(task_accuracy)

        argument_content_bias = func(task_group_accuracies)
        argument_content_biases.append(argument_content_bias)

    return np.mean(argument_content_biases)


def get_task_template_robustness(task_name: str, model_name: str, func: Callable = robustness_func):

    metrics = get_accuracy_and_certainty(task_name, model_name)
    accuracy_template_0 = metrics["template0"]["accuracy"]
    accuracy_template_1 = metrics["template1"]["accuracy"]
    accuracy_template_2 = metrics["template2"]["accuracy"]

    template_robustness = func([accuracy_template_0, accuracy_template_1, accuracy_template_2])

    return template_robustness


def get_template_robustness(model_name: str, func: Callable = robustness_func, task_names: list[str] = None):

    task_names = task_names or list(core_tasks)

    template_robustness_per_task = []
    for task_name in task_names:
        task_template_robustness = get_task_template_robustness(task_name, model_name, func=func)
        template_robustness_per_task.append(task_template_robustness)

    return np.mean(template_robustness_per_task)


def get_adjacent_tasks_pair_robustness(task_pair: tuple[str, str], model_name: str,
                                       func: Callable = robustness_func):
    task_pair_accuracies = []
    for task_name in task_pair:
        metrics = get_accuracy_and_certainty(task_name, model_name)
        task_accuracy = metrics["task"]["accuracy"]
        task_pair_accuracies.append(task_accuracy)

    robustness = func(task_pair_accuracies)

    return robustness


def get_adjacent_tasks_robustness(model_name: str, func: Callable = robustness_func,
                                  task_pairs: dict[tuple[str, str], str] = None):

    task_pairs = task_pairs or adjacent_tasks_tasks_pairs

    return np.mean([
        get_adjacent_tasks_pair_robustness(task_pair=task_pair, model_name=model_name, func=func)
        for task_pair in task_pairs
    ])


main_metric_name_to_metric = {
    "argument_order": get_argument_order_robustness,
    "argument_content": get_argument_content_robustness,
    "template": get_template_robustness,
    "adjacent_tasks": get_adjacent_tasks_robustness
}

adjacent_tasks_type_metric_name_to_metric = {
    "adjacent_by_definition":
        partial(get_adjacent_tasks_robustness, task_pairs=adjacent_tasks_by_definition_pairs),
    "adjacent_by_explicit_negation":
        partial(get_adjacent_tasks_robustness, task_pairs=adjacent_tasks_explicit_negation_pairs),
    "adjacent_by_character_awareness":
        partial(get_adjacent_tasks_robustness, task_pairs=adjacent_tasks_character_awareness_pairs),
    "adjacent_by_first_vs_last":
        partial(get_adjacent_tasks_robustness, task_pairs=adjacent_tasks_by_first_vs_last_pairs)
}

metric_name_to_metric = main_metric_name_to_metric | adjacent_tasks_type_metric_name_to_metric


def get_model_robustness(model_name, func: Callable = robustness_func):

    metrics = main_metric_name_to_metric.values()

    return np.mean(
        [metric(model_name=model_name, func=func) for metric in metrics]
    ).round(1)


def create_argument_order_robustness_csv(model_names: list[str] = None,
                                         task_names: list[str] = None,
                                         output_path=None
                                         ):
    model_names = model_names or list(paper_models)
    task_names = task_names or simple_answer_index_task_names

    rows: list[list] = list()
    first_row = [""]
    for task_name in task_names:
        first_row.extend([task_name] * 2)  # replicating the model name for easy conversion to a multiindex df
    rows.append(first_row)

    second_row = [""]
    second_row.extend(["acc ans is w1", "acc when ans is w2"] * len(task_names))
    rows.append(second_row)

    # the rest of the rows are the results
    for model_name in model_names:
        row = [model_name]
        for task_name in task_names:
            answer_index_metrics = get_simple_answer_index_bias(task_name, model_name)
            accuracy_index_0 = answer_index_metrics["task"]["accuracy_answer_index_0"]
            row.append(round(accuracy_index_0, 1))
            accuracy_index_1 = answer_index_metrics["task"]["accuracy_answer_index_1"]
            row.append(round(accuracy_index_1, 1))

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("argument_order_robustness.csv")

    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def create_template_robustness_csv(model_names: list[str] = None, task_names: list[str] = None,
                                   func: Callable = robustness_func, output_path=None):

    model_names = model_names or list(paper_models)
    task_names = task_names or list(core_tasks)

    rows: list[list] = list()

    first_row = [""] + [get_short_model_name(model_name) for model_name in model_names]  # shorten model names
    rows.append(first_row)

    # the rest of the rows are the results
    for task_name in task_names:
        row = list()
        row.append(task_name)

        for model_name in model_names:
            template_robustness = get_task_template_robustness(task_name, model_name, func=func)
            row.append(round(template_robustness, 1))

        rows.append(row)

    output_path = output_path or RESULTS_DIR.joinpath("template_robustness.csv")

    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)


def create_robustness_csv(model_names: list[str] = None, func: Callable = robustness_func,
                          include_mean=True, include_adjacent_tasks_breakdown=False,
                          output_path=None):

    model_names = model_names or list(paper_models)

    rows: list[list] = list()
    metric_names = list(main_metric_name_to_metric)
    if include_adjacent_tasks_breakdown:
        metric_names.extend(adjacent_tasks_type_metric_name_to_metric)

    first_row = [""] + metric_names
    rows.append(first_row)

    # the rest of the rows are the results
    for model_name in model_names:
        row = list()

        row.append(get_short_model_name(model_name))

        for metric_name in metric_names:
            metric = metric_name_to_metric[metric_name]
            robustness_value = metric(model_name=model_name, func=func)
            row.append(round(robustness_value, 1))

        rows.append(row)

    # create output path
    if include_adjacent_tasks_breakdown:
        filename = f"robustness_with_adjacent_tasks_breakdown.csv"
    else:
        filename = f"robustness.csv"
    output_path = output_path or RESULTS_DIR.joinpath(filename)

    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)

    if not include_mean:
        return

    # add a "mean" column as the first column
    df = pd.read_csv(output_path, header=0, index_col=0)
    main_metrics_df = df[list(main_metric_name_to_metric)]
    means_column = main_metrics_df.mean(axis="columns").round(1)
    df.insert(0, "mean", means_column)

    df.to_csv(output_path)


def homophones_accuracy_per_distractor_type(model_name="text-davinci-002"):
    task = core_tasks["homophones"]

    with open(task.default_data_path) as f_task_data:
        task_data = json.load(f_task_data)
    examples = task_data["examples"]

    predictions_path = task.predictions_dir.joinpath(f"{model_name}.json")
    with open(predictions_path) as f_predictions:
        predictions = json.load(f_predictions)

    results = {
        "easy_distractor_count": 0,
        "easy_distractor_score": 0,
        "semantic_distractor_count": 0,
        "semantic_distractor_score": 0
    }

    for id_ in predictions:

        type_ = examples[id_]["metadata"]["distractor_type"]
        results[f"{type_}_distractor_count"] += 1

        score = predictions[id_]["score"]
        results[f"{type_}_distractor_score"] += score

    print(f"easy count: {results['easy_distractor_count']}")
    print(f"easy acc: {results['easy_distractor_score'] / results['easy_distractor_count']}")
    print(f"semantic count: {results['semantic_distractor_count']}")
    print(f"semantic acc: {results['semantic_distractor_score'] / results['semantic_distractor_count']}")


def all_words_from_category_accuracy_per_num_distractors(subset_name="", model_name="text-davinci-002"):

    if subset_name:
        subset_name = f"_{subset_name}"

    task = all_tasks[f"all_words_from_category{subset_name}"]()

    with open(task.default_data_path) as f_task_data:
        task_data = json.load(f_task_data)
    examples = task_data["examples"]

    predictions_path = task.predictions_dir.joinpath(f"{model_name}.json")
    with open(predictions_path) as f_predictions:
        predictions = json.load(f_predictions)

    results = {
        "dist0_count": 0,
        "dist0_score": 0,
        "dist1_count": 0,
        "dist1_score": 0,
        "dist2_count": 0,
        "dist2_score": 0,
    }

    for id_ in predictions:

        n_dist = examples[id_]["metadata"]["num_distractors"]
        results[f"dist{n_dist}_count"] += 1

        score = predictions[id_]["score"]
        results[f"dist{n_dist}_score"] += score

    for i in [0, 1, 2]:
        print(f"dist{i} count: {results[f'dist{i}_count']}")
        print(f"dist{i} score: {results[f'dist{i}_score'] / (results[f'dist{i}_count'] or 0.0001)}")


def any_words_from_category_accuracy_per_num_distractors(model_name="text-davinci-002"):

    task = core_tasks[f"any_words_from_category"]()

    with open(task.default_data_path) as f_task_data:
        task_data = json.load(f_task_data)
    examples = task_data["examples"]

    predictions_path = task.predictions_dir.joinpath(f"{model_name}.json")
    with open(predictions_path) as f_predictions:
        predictions = json.load(f_predictions)

    results = {
        "dist4_count": 0,
        "dist4_score": 0,
        "dist5_count": 0,
        "dist5_score": 0,
    }

    for id_ in predictions:

        n_dist = examples[id_]["metadata"]["num_distractors"]
        results[f"dist{n_dist}_count"] += 1

        score = predictions[id_]["score"]
        results[f"dist{n_dist}_score"] += score

    for i in [4, 5]:
        print(f"dist{i} count: {results[f'dist{i}_count']}")
        print(f"dist{i} score: {results[f'dist{i}_score'] / (results[f'dist{i}_count'] or 0.0001)}")
