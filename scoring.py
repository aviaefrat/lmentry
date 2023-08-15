import argparse

from lmentry.analysis.accuracy import flexible_scoring
from lmentry.tasks.lmentry_tasks import all_tasks


def parse_arguments():
  parser = argparse.ArgumentParser(
    description="CLI for scoring all or specified LMentry predictions using the default file locations",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("-m", "--model_names", nargs="+", type=str, default=None,
                      help="Names of models which statistics were collected and should evaluate. "
                           "If it is None the predictions directory is analized and evailable model-statistics are evaluated")
  parser.add_argument('-t', '--task_names', nargs="+", type=str, default=None,
                      help="If need to evaluate specified set of tasks set their names. "
                           f"Task names should be from the list: {all_tasks.keys()}. "
                           "It tries to analyze all tasks by default")
  parser.add_argument("-n", "--num-procs", type=int, default=1,
                      help="The number of processes to use when scoring the predictions. "
                           "Can be up to the number of models you want to evaluate * 41.")
  return parser.parse_args()


def main():
  args = parse_arguments()

  flexible_scoring(task_names=args.task_names,
                   model_names=args.model_names,
                   num_processes=args.num_procs)


if __name__ == "__main__":
  main()
