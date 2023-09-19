import json
import random

from lmentry.constants import LMENTRY_WORDS_PATH
from lmentry.scorers.first_alphabetically_scorer import FirstAlphabeticallyScorer
from tasks.task import LMentryTask


class FirstAlphabetically(LMentryTask):

    scorer_cls = FirstAlphabeticallyScorer

    def __init__(self, name="first_alphabetically", words_path=None):
        super().__init__(name)
        self.canonical_template = "Q: In an alphabetical order, which of the words \"{word1}\" and \"{word2}\" comes first?\nA:"
        self.second_template = "Q: In an alphabetical order, which word comes first, \"{word1}\" or \"{word2}\"?\nA:"
        self.third_template = "Q: Of the words \"{word1}\" and \"{word2}\", which word comes first alphabetically?\nA:"
        self.parameter_names = ["answer", "distractor"]
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.words_path = words_path or LMENTRY_WORDS_PATH

    def get_task_words(self):
        with open(self.words_path) as f:
            task_words = json.load(f)
        for word in list(task_words):
            word_levels = set(task_words[word]["levels"])
            if not {"A1", "A2"} & word_levels:
                del task_words[word]

        return list(task_words)

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        task_words = self.get_task_words()

        # create examples
        examples = dict()
        for template_id, template in enumerate(self.all_templates):

            seed = seed or self.default_seed
            random.seed(seed)
            num_template_examples_created = 0
            already_seen_arguments: set[tuple[str, str]] = set()  # word1, word2

            while num_template_examples_created < num_examples:
                # sample the two words
                word1, word2 = random.sample(task_words, 2)

                if (word1, word2) in already_seen_arguments:
                    continue
                already_seen_arguments.add((word1, word2))

                input_ = template.format(word1=word1, word2=word2)
                answer_index = 0 if word1 < word2 else 1
                answer, distractor = (word1, word2) if answer_index == 0 else (word2, word1)

                # create the metadata
                metadata = dict()
                metadata["answer"] = answer
                metadata["distractor"] = distractor
                metadata["answer_index"] = answer_index
                metadata["word1"] = word1
                metadata["word2"] = word2
                metadata["template_id"] = template_id

                # create the example
                example = dict()
                example["input"] = input_
                example["metadata"] = metadata
                examples[str(len(examples) + 1)] = example

                num_template_examples_created += 1

        # create the shared settings
        settings = dict()
        settings["canary"] = self.canary_string
        settings["name"] = self.name
        settings["num_examples_per_template"] = num_examples
        settings["seed"] = seed
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data, task_data_path)


if __name__ == '__main__':
    task = FirstAlphabetically()
