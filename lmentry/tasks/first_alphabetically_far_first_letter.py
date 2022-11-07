import json
import random
import string
from collections import defaultdict

from lmentry.constants import LMENTRY_WORDS_PATH
from lmentry.scorers.first_alphabetically_scorer import FirstAlphabeticallyScorer
from lmentry.tasks.task import LMentryTask


class FirstAlphabeticallyFarFirstLetter(LMentryTask):

    scorer_cls = FirstAlphabeticallyScorer

    def __init__(self, name="first_alphabetically_far_first_letter", words_path=None):
        super().__init__(name)
        self.canonical_template = "Q: In an alphabetical order, which of the words \"{word1}\" and \"{word2}\" comes first?\nA:"
        self.second_template = "Q: In an alphabetical order, which word comes first, \"{word1}\" or \"{word2}\"?\nA:"
        self.third_template = "Q: Of the words \"{word1}\" and \"{word2}\", which word comes first alphabetically?\nA:"
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["answer", "distractor"]
        self.words_path = words_path or LMENTRY_WORDS_PATH

    def _create_input(self, answer: str, distractor: str, answer_index: int, template):

        if answer_index == 0:
            word1 = answer
            word2 = distractor
        elif answer_index == 1:
            word1 = distractor
            word2 = answer

        input_ = template.format(word1=word1, word2=word2)

        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        # get a dict of (<first letter>: <list of words word>) for all A1-A2 words
        with open(self.words_path) as f_words:
            words_data = json.load(f_words)
        a1_a2_words = []
        for word, word_data in words_data.items():
            if {"A1", "A2"} & set(word_data["levels"]):
                a1_a2_words.append(word)
        first_letter_to_words = defaultdict(list)
        for word in a1_a2_words:
            first_letter_to_words[word[0]].append(word)

        all_letters = string.ascii_lowercase
        all_answer_first_letter_options = [c for c in all_letters[:13] if c in first_letter_to_words]
        all_distractor_first_letter_options = [c for c in all_letters[13:] if c in first_letter_to_words]

        # create examples
        examples = dict()
        for template_id, template in enumerate(self.all_templates):

            seed = seed or self.default_seed
            random.seed(seed)
            num_template_examples_created = 0
            already_seen_arguments: set[tuple[str, str, int]] = set()  # answer, distractor, answer_index

            while num_template_examples_created < num_examples:
                sample_answer_first = random.choice((False, True))
                if sample_answer_first:
                    answer_first_letter = random.choice(all_answer_first_letter_options)
                    answer = random.choice(first_letter_to_words[answer_first_letter])

                    distractor_first_letter_options = [c for c in all_distractor_first_letter_options
                                                       if ord(c) - ord(answer_first_letter) >= 13]

                    if not distractor_first_letter_options:
                        continue
                    distractor_first_letter = random.choice(distractor_first_letter_options)
                    distractor = random.choice(first_letter_to_words[distractor_first_letter])

                else:  # sample distractor first
                    distractor_first_letter = random.choice(all_distractor_first_letter_options)
                    distractor = random.choice(first_letter_to_words[distractor_first_letter])

                    answer_first_letter_options = [c for c in all_answer_first_letter_options
                                                   if ord(distractor_first_letter) - ord(c) >= 13]

                    answer_first_letter = random.choice(answer_first_letter_options)
                    answer = random.choice(first_letter_to_words[answer_first_letter])

                answer_index = random.choice((0, 1))

                if (answer, distractor, answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((answer, distractor, answer_index))

                input_ = self._create_input(answer, distractor, answer_index, template)

                # create the metadata
                metadata = dict()
                metadata["answer"] = answer
                metadata["distractor"] = distractor
                metadata["answer_index"] = answer_index
                metadata["first_letter_distance"] = ord(distractor[0]) - ord(answer[0])
                metadata["word1"] = answer if answer_index == 0 else distractor
                metadata["word2"] = answer if answer_index == 1 else distractor
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
    task = FirstAlphabeticallyFarFirstLetter()
