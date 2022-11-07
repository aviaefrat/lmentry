import json
import random

from lmentry.constants import PHONETICALLY_UNAMBIGUOUS_WORDS_PATH, RHYME_GROUPS_PATH
from lmentry.scorers.rhyming_word_scorer import RhymingWordScorer
from lmentry.tasks.task import LMentryTask


class RhymingWord(LMentryTask):

    scorer_cls = RhymingWordScorer

    def __init__(self, name="rhyming_word"):
        super().__init__(name)
        self.canonical_template = "Q: Which word rhymes with the word \"{query}\", \"{word1}\" or \"{word2}\"?\nA:"
        self.second_template = "Q: Which is a rhyme of the word \"{query}\", \"{word1}\" or \"{word2}\"?\nA:"
        self.third_template = "Q: Of the words \"{word1}\" and \"{word2}\", which one rhymes with \"{query}\"?\nA:"
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["query", "answer", "distractor"]

    def create_data(self, num_examples=1000, seed=None,
                    rhyme_groups_path=None, distractors_path=None, task_data_path=None):

        # load rhyme groups
        rhyme_groups_path = rhyme_groups_path or RHYME_GROUPS_PATH
        with open(rhyme_groups_path) as f_rhyme_groups:
            rhyme_groups = json.load(f_rhyme_groups)

        # load the set of words from which we will select the distractors
        distractors_path = distractors_path or PHONETICALLY_UNAMBIGUOUS_WORDS_PATH
        with open(distractors_path) as f_distractors:
            distractors = set(json.load(f_distractors))  # we only need the words themselves, not their properties

        # create examples
        examples = dict()
        for template_id, template in enumerate(self.all_templates):

            seed = seed or self.default_seed
            random.seed(seed)
            num_template_examples_created = 0
            already_seen_arguments: set[tuple[str, str, str, int]] = set()  # query, answer, distractor, answer_index

            while num_template_examples_created < num_examples:

                # sample rhyme group proportionately to its size
                rhyme_group_names = list(rhyme_groups)
                rhyme_group_sizes = [len(rhyme_groups[name]) for name in rhyme_group_names]
                rhyme_group_name = random.sample(rhyme_group_names, k=1, counts=rhyme_group_sizes)[0]
                rhyme_group = rhyme_groups[rhyme_group_name]

                # sample query
                query = random.choice(rhyme_group)

                # sample answer
                # we don't want to have rhymes like "happy" and "unhappy"
                rhyme_options = [rhyme for rhyme in rhyme_group
                                 if not query.endswith(rhyme) and not rhyme.endswith(query)]
                # with the constraint above, we might not have any options left
                if len(rhyme_options) == 0:
                    continue
                answer = random.choice(rhyme_options)

                # sample distractor
                # make sure the distractor won't rhyme with the query
                distractor_options = list(distractors - set(rhyme_group))
                distractor_options.sort()  # to make the examples consistent across the same seed
                distractor = random.choice(distractor_options)

                # decide which of the answer and the distractor appears first
                answer_index = random.choice([0, 1])
                if (query, answer, distractor, answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((query, answer, distractor, answer_index))

                if answer_index == 0:
                    word1 = answer
                    word2 = distractor
                else:
                    word1 = distractor
                    word2 = answer

                input_ = template.format(query=query, word1=word1, word2=word2)

                # create the metadata
                metadata = dict()
                metadata["query"] = query
                metadata["answer"] = answer
                metadata["distractor"] = distractor
                metadata["answer_index"] = answer_index
                metadata["template_id"] = template_id

                # create the example
                example = dict()
                example["input"] = input_
                example["metadata"] = metadata
                examples[f"{len(examples) + 1}"] = example

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
    task = RhymingWord()
