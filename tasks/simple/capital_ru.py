from lmentry.scorers.simple_tasks_scorers import (
    CapitalRuScorer
)
from tasks.task import LMentryTask


capital_ru_QA_pairs = {
  "Франции": "Париж",
  "России": "Москва",
  "Великобритании": "Лондон",
}


class CapitalRu(LMentryTask):

  scorer_cls = CapitalRuScorer

  def __init__(self, name="capital_ru"):
      super().__init__(name)
      self.canonical_template = "Q: Как называется столица {country}?\nA:"
      self.second_template = "Q: Назовите столицу {country}\nA:"
      self.all_templates = [self.canonical_template, self.second_template]
      self.parameter_names = ["country", "answer"]

  def _create_input(self, template, country):
      input_ = template.format(country=country)
      return input_

  def create_data(self, task_data_path=None):
    examples = {}

    for template_id, template in enumerate(self.all_templates):
      num_examples_created = 0

      # create examples
      for country, answer in capital_ru_QA_pairs.items():
        num_examples_created += 1

        input_ = self._create_input(template, country)

        # create the metadata
        metadata = dict()
        metadata["country"] = country
        metadata["answer"] = answer
        metadata["template_id"] = template_id

        # create the example
        example = dict()
        example["input"] = input_
        example["metadata"] = metadata
        examples[str(len(examples) + 1)] = example

    # create the shared settings
    settings = dict()
    settings["canary"] = self.canary_string
    settings["name"] = self.name
    settings["num_examples_per_template"] = len(capital_ru_QA_pairs)
    settings["seed"] = None
    settings["input_templates"] = self.all_templates

    # build the task_data
    task_data = dict()
    task_data["settings"] = settings
    task_data["examples"] = examples

    # save the data
    self.save_task_data(task_data, task_data_path)


if __name__ == '__main__':
    task = CapitalRu()
    task.create_data()
