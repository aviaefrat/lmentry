import csv
from pathlib import Path
from typing import Callable

from lmentry.analysis.accuracy import get_model_accuracy
from lmentry.analysis.robustness import get_model_robustness
from lmentry.constants import paper_models, RESULTS_DIR, get_short_model_name


def create_lmentry_scores_csv(model_names: list[str] = None, model_shortname_func: Callable = None,
                              output_path: Path = None):

    model_names = model_names or list(paper_models)
    model_shortname_func = model_shortname_func or get_short_model_name

    rows: list[list] = []

    # first row
    rows.append(["", "LMentry Score", "Accuracy", "Robustness"])

    # rest of the rows
    for model_name in model_names:
        model_name_ = model_shortname_func(model_name)
        accuracy = get_model_accuracy(model_name)
        robustness = get_model_robustness(model_name)
        lmentry_score = accuracy * (robustness / 100)

        rows.append([model_name_, round(lmentry_score, 1), round(accuracy, 1), round(robustness, 1)])

    output_path = output_path or RESULTS_DIR.joinpath("lmentry_scores").with_suffix(".csv")

    with open(output_path, "w", newline="") as f_output:
        writer = csv.writer(f_output)
        writer.writerows(rows)

    print(f"LMentry scores saved in {output_path}")
