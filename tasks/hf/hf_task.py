from tasks.task import LMentryTask
from lmentry.constants import HF_TASKS_DATA_DIR


class HFTask(LMentryTask):
  CONFIG = None

  def __init__(self, name):
    super().__init__(name)

    # Create directory for HF tasks if need
    if not HF_TASKS_DATA_DIR.exists():
      HF_TASKS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    self.default_data_path = HF_TASKS_DATA_DIR.joinpath(f"{self.name}.json")

    self.DATASET_PATH = self.CONFIG.dataset_path if self.CONFIG else None
    self.DATASET_NAME = self.CONFIG.dataset_name if self.CONFIG else None
    # Prepare data in runtime when task is created
    self.create_data()

  def create_data(self, task_data_path=None):
    data_path = task_data_path or self.default_data_path
    if not data_path.exists():
      dataset = self.download_dataset() # HF Dataset object
      all_inputs_strs = dataset["context"]
      all_expectations = dataset["completion"]

      # create examples
      examples = {}

      for iid, question, answer in enumerate(zip(all_inputs_strs, all_expectations)):
        # create metadata
        metadata = dict()
        metadata["answer"]
        # create the example
        example = dict()
        example["input"] = question
        example["metadata"] = answer
        examples[f"{iid + 1}"] = example

      # create the shared settings
      settings = dict()
      settings["name"] = self.name

      # build the task_data
      task_data = dict()
      task_data["settings"] = settings
      task_data["examples"] = examples

      # save the data
      self.save_task_data(task_data, task_data_path)

  def download_dataset(self):
    import datasets
    return datasets.load_dataset(
      path=self.DATASET_PATH,
      name=self.DATASET_NAME,
    )
