import string

from lmentry.scorers.word_containing_scorer import WordContainingScorer
from lmentry.tasks.task import LMentryTask


class WordContaining(LMentryTask):

    scorer_cls = WordContainingScorer

    def __init__(self, name="word_containing"):
        super().__init__(name)
        self.canonical_template = 'Write a word that contains the letter "{letter}":\n'
        self.second_template = 'Write a word using the letter "{letter}":\n'
        self.third_template = 'Write a word containing the letter "{letter}":\n'
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
    task = WordContaining()
