import string

from lmentry.scorers.starts_with_letter_scorer import StartsWithLetterScorer
from lmentry.tasks.task import LMentryTask


class StartsWithLetter(LMentryTask):

    scorer_cls = StartsWithLetterScorer

    def __init__(self, name="starts_with_letter"):
        super().__init__(name)

        self.canonical_template = 'Write a word that starts with the letter "{letter}":\n'
        self.second_template = 'Write a word whose first letter is "{letter}":\n'
        self.third_template = 'Write a word starting with "{letter}":\n'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["letter"]

    def _create_input(self, template, letter):
        input_ = template.format(letter=letter)
        return input_

    def create_data(self, task_data_path=None):

        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):
            for letter in string.ascii_lowercase:
                # create the input
                input_ = self._create_input(template, letter)

                # create the metadata
                metadata = dict()
                metadata["letter"] = letter
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
        settings["num_examples_per_template"] = len(string.ascii_lowercase)
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data, task_data_path)


if __name__ == '__main__':
    task = StartsWithLetter()
