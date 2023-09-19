import pandas as pd

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.last_word_scorer import LastWordScorer
from tasks.task import LMentryTask


class LastWord(LMentryTask):

    scorer_cls = LastWordScorer

    def __init__(self, name="last_word"):
        super().__init__(name)

        self.canonical_template = 'Q: What is the last word of the sentence "{sentence}"?\nA:'
        self.second_template = 'Q: In the sentence "{sentence}", what is the last word?\nA:'
        self.third_template = 'Write the last word of the sentence "{sentence}":\n'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["sentence", "answer"]  # sentence is first because it includes the answer

    def _create_input(self, template, sentence):
        input_ = template.format(sentence=sentence)
        return input_

    def create_data(self, task_data_path=None):
        with open(RESOURCES_DIR.joinpath("simple_sentences.csv")) as f:
            source_data = pd.read_csv(f)
        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):
            for _, row in source_data.iterrows():
                # create the input
                sentence = row["Sentence"]

                # We remove the period at the end of sentences. In our sentences, period can only be at the end of
                # the sentence. Sentences end with or without a period, but can't end with any other non-alpha char.
                sentence = sentence.replace(".", "")
                input_ = self._create_input(template, sentence)

                # create the metadata
                metadata = dict()
                metadata["sentence"] = sentence
                metadata["sentence_source"] = row["Source"]
                # our sentences do not contain punctuation, so we can use space tokenization to get the last word
                metadata["answer"] = sentence.split()[-1].replace(".", "")
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
        settings["num_examples_per_template"] = len(list(source_data["Sentence"]))
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data, task_data_path)


if __name__ == '__main__':
    task = LastWord()
