import json
import random
from collections import Counter

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.less_letters_scorer import LessLettersScorer
from tasks.task import LMentryTask


class LessLetters(LMentryTask):

    scorer_cls = LessLettersScorer

    def __init__(self, name="less_letters"):
        super().__init__(name)

        self.canonical_template = 'Q: Which word has fewer letters, "{word1}" or "{word2}"?\nA:'
        self.second_template = 'Q: Which word is shorter, "{word1}" or "{word2}"?\nA:'
        self.third_template = 'Q: Of the words "{word1}" and "{word2}" which one has fewer letters?\nA:'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["answer", "distractor"]

    def get_source_data(self):
        words = {}
        with open(RESOURCES_DIR.joinpath("lmentry_words.json")) as f:
            source_data = json.load(f)
        for word, word_info in source_data.items():
            if word.lower() == "a":
                continue
            if "A1" in word_info["levels"] or "A2" in word_info["levels"]:
                # We use words that have all unique letters
                word_letters_counter = Counter(word)
                most_common_letter_count = word_letters_counter.most_common(1)[0][1]
                if most_common_letter_count == 1:
                    words[word] = word_info
        return words

    def _create_input(self, template, word1, word2):
        input_ = template.format(word1=word1, word2=word2)
        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):
        source_data = self.get_source_data()
        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):
            # set random seed
            seed = seed or self.default_seed
            random.seed(seed)
            num_examples_created = 0
            already_sampled = []

            # create examples
            while num_examples_created < num_examples:
                # sample the words
                sampled_words = random.sample(list(source_data), 2)
                if sampled_words in already_sampled or len(sampled_words[0]) == len(sampled_words[1]):
                    continue
                already_sampled.append(sampled_words)
                num_examples_created += 1

                # create the input
                input_ = self._create_input(template, sampled_words[0], sampled_words[1])
                words_lengths = [len(sampled_words[0]), len(sampled_words[1])]
                shorter_word_idx = words_lengths.index(min(words_lengths))

                # create the metadata
                metadata = dict()
                metadata["word1"] = sampled_words[0]
                metadata["word2"] = sampled_words[1]
                metadata["answer"] = sampled_words[shorter_word_idx]
                metadata["distractor"] = sampled_words[abs(1-shorter_word_idx)]
                metadata["answer_index"] = shorter_word_idx
                metadata["word_lengths"] = [len(sampled_words[0]), len(sampled_words[1])]
                metadata["word_pos_tags"] = [source_data[sampled_words[0]]["pos"],
                                              source_data[sampled_words[1]]["pos"]]
                metadata["word_levels"] = [source_data[sampled_words[0]]["levels"],
                                            source_data[sampled_words[1]]["levels"]]
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
        settings["seed"] = seed
        settings["num_examples_per_template"] = num_examples
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data, task_data_path)


if __name__ == '__main__':
    task = LessLetters()
