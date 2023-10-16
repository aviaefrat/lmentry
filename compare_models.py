import argparse
import logging
from tqdm import tqdm

from lmentry.constants import PREDICTIONS_ROOT_DIR, TASKS_DATA_DIR, RESULTS_DIR, DEFAULT_MAX_LENGTH
from tasks.task_utils import get_tasks_names, task_groups, all_tasks
from lmentry.predict import generate_all_hf_predictions
from lmentry.analysis.accuracy import flexible_scoring
from lmentry.analysis.comparison import create_per_task_accuracy_comparison_csv
from lmentry.model_manager import get_short_model_names


def parse_arguments():
  parser = argparse.ArgumentParser(
      description="CLI for comparison of two models (reference and probe) using LMentry tasks from the default file locations and additional heuristics",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("-r", "--ref_model_name", type=str, default="vicuna-7b-v1-3-q0f16",
                      help="Name of reference model. It is assumed that the model is original, "
                           "uses high-precision data type and has better accuracy")
  parser.add_argument('-p', '--probe_model_names', nargs="+", type=str, default=["vicuna-7b-v1-3-q4f16_0"],
                      help=f"Names of probe models. If the number of the probe models is bigger than one "
                           "it iteratively compares the reference model with each from the list.")
  parser.add_argument('-t', '--task_names', nargs="+", type=str, default=get_tasks_names("7b"),
                      help="If need to evaluate specified set of tasks set their names or name(s) of specified task set(s). "
                           f"Task set names should be from the list: {task_groups.keys()}. "
                           f"Task names should be from the list: {all_tasks}. "
                           "It tries to analyze 7b-model sensetive task set by default")
  parser.add_argument('-d', '--device', type=str, default="cuda",
                      help="Device name. It is needed and used by mlc model only during predictions")
  parser.add_argument('-b', '--batch_size', type=int, default=100,
                      help="For calculation on A10G batch size 100 is recommended. "
                           "For mlc-llm models batch size is reduced to 1. "
                           "It is used duirng predictions.")
  parser.add_argument('-s', '--samples_num', type=int, default=None,
                      help="Number of samples to choose randomly from task dataset. "
                           "If set 'None' or the value is bigger than all samples number - all samples will be chosen.")
  parser.add_argument('-ml', '--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                      help="Input max length. It is used duirng predictions.")
  parser.add_argument("-n", "--num-procs",
                      default=1,
                      type=int,
                      help="The number of processes to use when scoring the predictions. "
                           "Can be up to the number of models you want to evaluate * 41.")
  parser.add_argument("-f", "--forced_scoring", action="store_true", default=False,
                      help="If scoring has been done for specified task it skips it. This flag allows to redo ready scoring")
  parser.add_argument("-c", "--certainty", action="store_true", default=False,
                      help="Conservative accuracy evaluation. The answer is considered correct only if it is absolutely certain")
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
  task_names = get_tasks_names(args.task_names)

  for probe_model_name in tqdm(args.probe_model_names, desc="Models comparison"):
    model_names = get_short_model_names([args.ref_model_name, probe_model_name])
    print(f"Models {model_names[0]} and {model_names[1]} are compared")

    # Predict specified tasks for given models
    # Reference model
    logging.info(f"Prediction for {model_names[0]} model starts")
    generate_all_hf_predictions(
      task_names=task_names,
      model_name=args.ref_model_name,
      max_length=args.max_length,
      batch_size=args.batch_size,
      device=args.device,
      samples_num=args.samples_num,
    )
    logging.info(f"Prediction for {model_names[0]} model finished")
    # Probe_model
    logging.info(f"Prediction for {model_names[1]} model starts")
    generate_all_hf_predictions(
      task_names=task_names,
      model_name=probe_model_name,
      max_length=args.max_length,
      batch_size=args.batch_size,
      device=args.device,
      samples_num=args.samples_num,
    )
    logging.info(f"Prediction for {model_names[1]} model finished")

    flexible_scoring(
      task_names=task_names,
      model_names=model_names,
      num_processes=args.num_procs,
      forced_scoring=args.forced_scoring,
    )

    create_per_task_accuracy_comparison_csv(model_names=model_names, task_names=task_names, certainty=args.certainty)


if __name__ == "__main__":
  main()
