import json
import random
from collections import defaultdict

from lmentry.constants import LMENTRY_WORDS_PATH
from lmentry.scorers.first_alphabetically_scorer import FirstAlphabeticallyScorer
from lmentry.tasks.task import LMentryTask


class FirstAlphabeticallyDifferentFirstLetter(LMentryTask):

    scorer_cls = FirstAlphabeticallyScorer

    def __init__(self, name="first_alphabetically_different_first_letter", words_path=None):
        super().__init__(name)
        self.canonical_template = "Q: In an alphabetical order, which of the words \"{word1}\" and \"{word2}\" comes first?\nA:"
        self.second_template = "Q: In an alphabetical order, which word comes first, \"{word1}\" or \"{word2}\"?\nA:"
        self.third_template = "Q: Of the words \"{word1}\" and \"{word2}\", which word comes first alphabetically?\nA:"
        self.parameter_names = ["answer", "distractor"]
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.words_path = words_path or LMENTRY_WORDS_PATH

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
        # remove entries where a letter has no words starting with it
        first_letter_to_words = {letter: words for letter, words in first_letter_to_words.items()}
        first_letter_options = list(first_letter_to_words)

        # create examples
        examples = dict()
        for template_id, template in enumerate(self.all_templates):

            seed = seed or self.default_seed
            random.seed(seed)
            num_template_examples_created = 0
            already_seen_arguments: set[tuple[str, str, int]] = set()  # word1, word2, answer_index

            while num_template_examples_created < num_examples:
                # sample the two words
                first_letter_word1, first_letter_word2 = random.sample(first_letter_options, 2)
                word1 = random.choice(first_letter_to_words[first_letter_word1])
                word2 = random.choice(first_letter_to_words[first_letter_word2])

                answer_index = 0 if word1 < word2 else 1

                # we can already check for duplicates, but I'm determining the answer and distractor
                # to keep it consistent with other "first alphabetically" tasks
                answer, distractor = (word1, word2) if word1 < word2 else (word2, word1)
                if (answer, distractor, answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((answer, distractor, answer_index))

                input_ = template.format(word1=word1, word2=word2)

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
    task = FirstAlphabeticallyDifferentFirstLetter()
