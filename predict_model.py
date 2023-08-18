import argparse

from lmentry.predict import generate_all_hf_predictions
from lmentry.tasks.lmentry_tasks import all_tasks
from lmentry.constants import DEFAULT_MAX_LENGTH


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model_name', type=str, default="vicuna-7b-v1-3",
                      help="Model name or path to the root directory of mlc-llm model")
  parser.add_argument('-t', '--task_name', type=str, default=None,
                      help=f"If need to predict only one task set its name. Name should be from the list: {all_tasks.keys()}")
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

  task_names = None
  if args.task_name is not None:
    task_names = [args.task_name]

  generate_all_hf_predictions(
    task_names=task_names,
    model_name=args.model_name,
    max_length=args.max_length,
    batch_size=args.batch_size,
    device=args.device,
  )


if __name__ == "__main__":
  main()
