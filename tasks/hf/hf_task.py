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

    if self.CONFIG is None:
      raise NotImplementedError("HFTask is abstract class which does not work without config which defined in child one")
    self.DATASET_PATH = self.CONFIG.dataset_path
    self.DATASET_NAME = self.CONFIG.dataset_name
    # Prepare data in runtime when task is created
    self.create_data()

  def create_data(self, task_data_path=None):
    data_path = task_data_path or self.default_data_path
    if not data_path.exists():
      dataset = self.get_dataset_from_hf() # HF Dataset object
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

  def get_dataset_from_hf(self):
    dataset = self.download_dataset()
    if self.has_test_docs():
        if self.CONFIG.process_docs is not None:
          return self.CONFIG.process_docs(dataset[self.CONFIG.test_split])
        return dataset[self.CONFIG.test_split]
    elif self.has_validation_docs():
      if self.CONFIG.process_docs is not None:
        return self.CONFIG.process_docs(dataset[self.CONFIG.validation_split])
      return dataset[self.CONFIG.validation_split]
    else:
        raise ImportError(f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!")

  def download_dataset(self):
    import datasets
    kwargs = self.CONFIG.dataset_kwargs
    return datasets.load_dataset(
      path=self.DATASET_PATH,
      name=self.DATASET_NAME,
      **kwargs if kwargs is not None else {},
    )

  def has_validation_docs(self) -> bool:
    if self.CONFIG.validation_split is not None:
      return True
    else:
      return False

  def has_test_docs(self) -> bool:
    if self.CONFIG.test_split is not None:
      return True
    else:
      return False
