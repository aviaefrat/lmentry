import json
import random
from collections import defaultdict, Counter

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.more_letters_scorer import MoreLettersScorer
from tasks.task import LMentryTask


class MoreLettersLengthDiff3plus(LMentryTask):

    scorer_cls = MoreLettersScorer

    def __init__(self, name="more_letters_length_diff_3plus"):
        super().__init__(name)

        self.canonical_template = 'Q: Which word has more letters, "{word1}" or "{word2}"?\nA:'
        self.second_template = 'Q: Which word is longer, "{word1}" or "{word2}"?\nA:'
        self.third_template = 'Q: Of the words "{word1}" and "{word2}" which one has more letters?\nA:'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["answer", "distractor"]
        self.num_to_word = dict(
            zip(range(2, 11), ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])
        )

    def get_source_data(self):
        # get a dict of (<word length>: <list of words>) for all A1-A2 words
        length_to_words = defaultdict(list)
        with open(RESOURCES_DIR.joinpath("lmentry_words.json")) as f:
            source_data = json.load(f)
        for word, word_info in source_data.items():
            if word.lower() == 'a':
                continue
            if "A1" in word_info["levels"] or "A2" in word_info["levels"]:
                # We use words that have all unique letters
                word_letters_counter = Counter(word)
                most_common_letter_count = word_letters_counter.most_common(1)[0][1]
                if most_common_letter_count == 1:
                    word_info["word"] = word
                    length_to_words[len(word)].append(word_info)
        return length_to_words

    def _create_input(self, template, answer, distractor, answer_index):
        if answer_index == 0:
            input_ = template.format(word1=answer, word2=distractor)
        else:
            input_ = template.format(word1=distractor, word2=answer)
        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        source_data = self.get_source_data()

        lengths = sorted(source_data)

        min_length = lengths[0]
        min_answer_length = min([length for length in lengths if length >= min_length + 3])
        all_answer_length_options = lengths[lengths.index(min_answer_length):]

        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):
            # set random seed
            seed = seed or self.default_seed
            random.seed(seed)
            num_examples_created = 0
            already_seen_arguments: set[tuple[str, str, int]] = set()  # answer, distractor, answer_index

            # create examples
            while num_examples_created < num_examples:
                answer_length = random.choice(all_answer_length_options)
                answer = random.choice(source_data[answer_length])
                distractor_options = []
                for l in range(min_length, answer_length-2):
                    if l in source_data:
                        distractor_options.extend(source_data[l])
                if not distractor_options:
                    continue
                distractor = random.choice(distractor_options)

                answer_index = random.choice((0, 1))

                answer_word = answer["word"]
                distractor_word = distractor["word"]
                if (answer_word, distractor_word, answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((answer_word, distractor_word, answer_index))
                num_examples_created += 1

                # create the input
                input_ = self._create_input(template, answer_word, distractor_word, answer_index)
                words_by_order = [answer, distractor] if answer_index == 0 else [distractor, answer]

                # create the metadata
                metadata = dict()
                metadata["word1"] = words_by_order[0]["word"]
                metadata["word2"] = words_by_order[1]["word"]
                metadata["answer"] = answer_word
                metadata["distractor"] = distractor_word
                metadata["answer_index"] = answer_index
                metadata["word_lengths"] = [len(words_by_order[0]["word"]), len(words_by_order[1]["word"])]
                metadata["word_pos_tags"] = [words_by_order[0]["pos"], words_by_order[1]["pos"]]
                metadata["word_levels"] = [words_by_order[0]["levels"], words_by_order[1]["levels"]]
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
    task = MoreLettersLengthDiff3plus()
