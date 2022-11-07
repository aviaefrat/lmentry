import json
import random

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.least_associated_word_scorer import LeastAssociatedWordScorer
from lmentry.tasks.task import LMentryTask


class LeastAssociatedWord(LMentryTask):

    scorer_cls = LeastAssociatedWordScorer

    def __init__(self, name="least_associated_word"):
        super().__init__(name)

        self.canonical_template = \
            'Q: Of the words {options}, what is the word least associated with "{category}"?\nA:'
        self.second_template = 'Q: What is the word least related to the word "{category}" from the words {options}?\nA:'
        self.third_template = \
            'Of the following words, choose the word least associated with the word "{category}" - {options}:\n'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.non_and_template_id = 2
        self.parameter_names = ["category", "answer"]

    def get_source_data(self):

        with open(RESOURCES_DIR.joinpath("nouns-by-category.json")) as f:
            source_data = json.load(f)
        source_data["clothes"] = source_data.pop("items of clothing")
        return source_data

    def _create_input(self, template, category, distractors, answer, answer_index, template_id):

        # create the options from the distractors and the answer
        options = distractors.copy()
        options.insert(answer_index, answer)

        # create the string representation of the options
        options = [f"\"{option}\"" for option in options]
        if template_id != self.non_and_template_id:
            options = ", ".join(options[:-1]) + f", and {options[-1]}"
        else:
            options = ", ".join(options)
        input_ = template.format(category=category, options=options)

        return input_

    def create_data(self, num_distractors_range: tuple = (4,), num_examples=1000, seed=None,
                    task_data_path=None):
        source_data = self.get_source_data()
        # create examples
        examples = {}
        categories = list(source_data.keys())
        for template_id, template in enumerate(self.all_templates):
            # set random seed
            seed = seed or self.default_seed
            random.seed(seed)

            num_template_examples_created = 0
            already_seen_arguments = set()

            while num_template_examples_created < num_examples:
                answer_category, distractor_category = random.sample(categories, 2)

                # sample answer
                answer = random.choice(source_data[answer_category])
                num_distractors = random.choice(num_distractors_range)
                # sample category and distractors
                distractors = random.sample(source_data[distractor_category],
                                            num_distractors)
                category = distractor_category

                answer_index = random.choice(range(num_distractors + 1))
                if (category, answer, tuple(distractors), answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((category, answer, tuple(distractors), answer_index))

                # create the input
                input_ = self._create_input(template, category, distractors, answer, answer_index,
                                            template_id=template_id)
                # create the metadata
                metadata = dict()
                metadata["category"] = category
                metadata["answer"] = answer
                metadata["distractors"] = distractors
                metadata["distractor_category"] = distractor_category
                metadata["answer_category"] = answer_category
                metadata["answer_index"] = answer_index
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
        settings["num_distractors_range"] = num_distractors_range
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
    task = LeastAssociatedWord()
