import argparse
import logging
from collections import defaultdict
from pathlib import Path

from lmentry.analysis.accuracy import score_all_predictions, create_per_task_accuracy_csv, \
    create_per_template_accuracy_csv
from lmentry.analysis.lmentry_score import create_lmentry_scores_csv
from lmentry.analysis.robustness import create_robustness_csv, create_argument_order_robustness_csv
from lmentry.constants import PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR, RESULTS_DIR
from lmentry.tasks.lmentry_tasks import all_tasks

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CLI for evaluating LMentry using the default file locations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-procs",
                        default=1,
                        type=int,
                        help="the number of processes to use when scoring the predictions.\n"
                             "can be up to the number of models you want to evaluate * 41."
                        )
    return parser.parse_args()


def check_for_missing_predictions(predictions_root_dir: Path):
    # predictions_root_dir should include exactly 41 directories, each one named after one of the 25
    # "core" tasks + the 16 "argument content robustness" tasks
    existing_predictions = defaultdict(set)

    for task_predictions_dir in predictions_root_dir.iterdir():
        if not task_predictions_dir.is_dir():
            continue
        task_name = task_predictions_dir.name
        for model_predictions_path in task_predictions_dir.iterdir():
            model_name = model_predictions_path.stem.replace("_with_metadata", "")
            existing_predictions[model_name].add(task_name)

    all_task_names = set(all_tasks)
    missing_predictions = dict()
    for model_name in existing_predictions:
        missing_predictions_for_model = sorted(all_task_names - existing_predictions[model_name])
        missing_predictions[model_name] = missing_predictions_for_model

    return missing_predictions


def main():
    if not TASKS_DATA_DIR.exists():
        logging.error(f"LMentry tasks data not found at {TASKS_DATA_DIR}. aborting.\n")
        return
    if not PREDICTIONS_ROOT_DIR.exists():
        logging.error(f"Predictions not found at {PREDICTIONS_ROOT_DIR}. aborting.\n")
        return
    RESULTS_DIR.mkdir(exist_ok=True)

    missing_predictions = check_for_missing_predictions(PREDICTIONS_ROOT_DIR)
    if any([len(missing_predictions[model_name]) > 0 for model_name in missing_predictions]):
        missing_predictions_str = "\n".join([f"{model_name}: {task_names}" for
                                             model_name, task_names in
                                             sorted(missing_predictions.items())])
        logging.error("Some models don't have all task predictions. aborting.\n"
                      "missing predictions (<model>: <predictions>):\n"
                      f"{missing_predictions_str}\n"
                      "for scoring an individual task, see `score_task_predictions` from `accuracy.py`.")
        return

    args = parse_arguments()
    model_names = sorted(missing_predictions.keys())
    logging.info(f"scoring LMentry predictions for models {sorted(missing_predictions.keys())}")
    score_all_predictions(task_names=sorted(all_tasks.keys()),
                          model_names=model_names,
                          num_processes=args.num_procs
                          )
    logging.info(f"finished scoring all LMentry predictions for models {model_names}")

    create_per_task_accuracy_csv(model_names=model_names)
    create_per_template_accuracy_csv(model_names=model_names)
    create_robustness_csv(model_names)
    create_lmentry_scores_csv(model_names)
    create_argument_order_robustness_csv(model_names=model_names)
    # todo add argument content robustness csv



if __name__ == "__main__":
    main()
