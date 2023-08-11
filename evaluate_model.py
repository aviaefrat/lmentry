import argparse
import logging

from lmentry.constants import PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR, RESULTS_DIR
from lmentry.analysis.accuracy import (
  score_all_predictions,
  create_per_task_accuracy_csv,
  create_per_template_accuracy_csv,
)


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="CLI for evaluating LMentry using the default file locations",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('-m', '--model_name', type=str, default="vicuna-7b-v1-3",
                      help="Model name or path to the root directory of mlc-llm model")
  parser.add_argument("--num-procs",
                      default=1,
                      type=int,
                      help="the number of processes to use when scoring the predictions.\n"
                            "can be up to the number of models you want to evaluate * 41.")
  return parser.parse_args()


def main():
  if not TASKS_DATA_DIR.exists():
    logging.error(f"LMentry tasks data not found at {TASKS_DATA_DIR}. aborting.\n")
    return
  if not PREDICTIONS_ROOT_DIR.exists():
    logging.error(f"Predictions not found at {PREDICTIONS_ROOT_DIR}. aborting.\n")
    return
  RESULTS_DIR.mkdir(exist_ok=True)

  args = parse_arguments()
  # TODO(vchernov): hardcode to quick check
  task_names = ["first_letter"]
  model_names = ["vicuna-7b-v1-3", "vicuna7bv1", "vicuna-7b-v1-3-q4f16_0"] # HF, mlc_q0f16, mlc_q4f16_0
  logging.info(f"scoring LMentry predictions for models {model_names}")
  score_all_predictions(task_names=task_names,
                        model_names=model_names,
                        num_processes=args.num_procs
                        )
  logging.info(f"finished scoring all LMentry predictions for models {model_names}")

  create_per_task_accuracy_csv(task_names=task_names, model_names=model_names)
  create_per_template_accuracy_csv(task_names=task_names, model_names=model_names)


if __name__ == "__main__":
  main()
