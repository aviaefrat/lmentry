import argparse
from tqdm import tqdm

from lmentry.predict import generate_all_hf_predictions
from lmentry.tasks.lmentry_tasks import get_tasks_names, tasks_list
from lmentry.constants import DEFAULT_MAX_LENGTH


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_names", nargs="+", type=str, default="vicuna-7b-v1-3",
                      help="Model names or paths to the root directory of mlc-llm models for predictions.")
  parser.add_argument('-t', '--task_names', nargs="+", type=str, default=get_tasks_names(),
                      help="If need to evaluate specified set of tasks set their names or name(s) of specified task set(s). "
                           f"Task set names should be from the list: {tasks_list.keys()}. "
                           f"Task names should be from the list: {get_tasks_names()}. "
                           "It tries to analyze all tasks by default")
  parser.add_argument('-d', '--device', type=str, default="cuda",
                      help="Device name. It is needed and used by mlc model only")
  parser.add_argument('-b', '--batch_size', type=int, default=100,
                      help="For calculation on A10G batch size 100 is recommended. For mlc-llm models batch size is reduced to 1")
  parser.add_argument('-ml', '--max_length', type=int, default=DEFAULT_MAX_LENGTH,
                      help="Input max length")

  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  task_names = get_tasks_names(args.task_names)

  for model_name in tqdm(args.model_names, desc="Predict specified models"):
    print(f"Prediction of tasks for {model_name} model starts")
    generate_all_hf_predictions(
      task_names=task_names,
      model_name=model_name,
      max_length=args.max_length,
      batch_size=args.batch_size,
      device=args.device,
    )
    print(f"Prediction of tasks for {model_name} model finished")


if __name__ == "__main__":
  main()
