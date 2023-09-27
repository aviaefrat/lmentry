import collections
import random

import nltk
import pandas as pd
from nltk.corpus import stopwords

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.word_after_scorer import WordAfterScorer
from tasks.task import LMentryTask


class WordAfter(LMentryTask):

    scorer_cls = WordAfterScorer

    def __init__(self, name="word_after"):
        super().__init__(name)

        self.canonical_template = 'Q: In the sentence "{sentence}", which word comes right after the word "{word}"?\nA:'
        self.second_template = 'Q: In the sentence "{sentence}", which word immediately succeeds the word "{word}"?\nA:'
        self.third_template = 'Q: Which word comes right after "{word}" in the sentence "{sentence}"?\nA:'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["sentence", "query", "answer"]  # sentence is first because it includes the others

    def _create_input(self, template, sentence, word):
        input_ = template.format(sentence=sentence, word=word)
        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        with open(RESOURCES_DIR.joinpath("simple_sentences.csv")) as f:
            source_data = pd.read_csv(f)

        try:  # using a try block because nltk.download hangs too many times
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))

        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):

            all_possible_positions = []

            # set random seed
            seed = seed or self.default_seed
            random.seed(seed)

            for _, row in source_data.iterrows():
                sentence = row["Sentence"]

                # We remove the period at the end of sentences. In our sentences, period can only be at the end of
                # the sentence. Sentences end with or without a period, but can't end with any other non-alpha char.
                sentence = sentence.replace(".", "")

                # our sentences do not contain punctuation, so we can use space tokenization to get the last word
                sentence_tokenized = sentence.replace('.', '').split()
                sentence_lower_tokenized = sentence.lower().replace('.', '').split()
                word_counter = collections.Counter(sentence_lower_tokenized)

                for i, word in enumerate(sentence_tokenized[:-1]):
                    # when asking about the word after, we do not ask about the last word of the sentence.
                    if word_counter[word.lower()] > 1:
                        # we avoid asking about words that appear more than once in the sentence.
                        continue
                    if sentence_lower_tokenized[i+1] in self.stopwords:
                        # we avoid examples where the answer is a stopword (e.g. "the") as this makes
                        # it harder for an accurate automatic evaluation
                        continue

                    # create the input
                    input_ = self._create_input(template, sentence, word)

                    # create the metadata
                    metadata = dict()
                    metadata["sentence"] = sentence
                    metadata["sentence_source"] = row["Source"]
                    metadata["query"] = word
                    metadata["query_position_in_sentence"] = i
                    metadata["answer"] = sentence_tokenized[i+1]
                    metadata["template_id"] = template_id

                    # create the example
                    example = dict()
                    example["input"] = input_
                    example["metadata"] = metadata
                    all_possible_positions.append(example)

            # sample final examples
            sampled_examples = random.sample(all_possible_positions, num_examples)
            for example in sampled_examples:
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
    task = WordAfter()
