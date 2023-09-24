import yaml
import fnmatch
from pathlib import Path


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
  if type(patterns) == str:
    patterns = [patterns]

  task_names = set()
  for pattern in patterns:
    for matching in fnmatch.filter(source_list, pattern):
      task_names.add(matching)
  return sorted(list(task_names))


def load_yaml_config(yaml_path: Path):
  with yaml_path.open("rb") as file:
    yaml_config = yaml.full_load(file)
    yaml_dir = yaml_path.parent

    if "include" in yaml_config:
      include_path = yaml_config["include"]
      del yaml_config["include"]

      if type(include_path) == str:
          include_path = [include_path]

      # Load from the last one first
      include_path.reverse()
      final_yaml_config = {}
      for path_str in include_path:
        # Assumes that path is a full path.
        # If not found, assume the included yaml
        # is in the same dir as the original yaml
        path = Path(path_str)
        if not path.is_file():
          path = yaml_dir.joinpath(path_str)

        try:
            included_yaml_config = load_yaml_config(path)
            final_yaml_config.update(included_yaml_config)
        except Exception as ex:
            # If failed to load, ignore
            raise ex

      final_yaml_config.update(yaml_config)
      return final_yaml_config
    return yaml_config

