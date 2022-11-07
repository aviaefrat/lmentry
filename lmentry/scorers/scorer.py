import itertools
import json
import logging
import re
from collections import Counter
from pathlib import Path

from num2words import num2words

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)


word_regex_template = r'(The word )?["\'\.,]*\b{}\b["\'\.,]*'
sentence_regex_template = r'((The sentence )?["\'\.,]*{}["\'\.,]*|The sentence)'
letter_regex_template = r'(The letter )?["\'\.,]*\b{}\b["\'\.,]*'
number_regex_template = r'(The number )?["\'\.,]*\b{}\b["\'\.,]*'


def standardized_number_regex(number: int):
    digits_string = str(number)
    word_form = num2words(number)
    return fr'(\b{digits_string}\b|{word_form}|{word_form.replace("-", " ")})'


def the_number_regex(number: int):
    standardized_number_regex_ = standardized_number_regex(number)
    return number_regex_template.format(standardized_number_regex_)


def the_word_regex(word: str):
    return word_regex_template.format(word)


def the_words_regex(words: list[str]):
    # we use `[^a-zA-Z0-9_]` instead of `\W` as we always lowercase the pattern in
    # `certainty_scorer`. See the to-do suggestion in normalize_string
    W = r"[^a-zA-Z0-9_]"

    words_regex = rf"{W}*"
    for i, word in enumerate(words):
        words_regex += rf"{word}{W}+"
        if i == len(words) - 2:
            words_regex += rf"(and{W}+)?"
    words_regex += rf"{W}*"

    return rf"(The words )?{words_regex}"


def the_list_regex(words: list[str]):

    words_as_string = '", "'.join(words)
    words_as_string = rf'\["{words_as_string}"]'

    return rf"(((The list )?{words_as_string})|the list)"


def the_sentence_regex(sentence: str):
    return sentence_regex_template.format(sentence)


def the_letter_regex(letter: str):
    return letter_regex_template.format(letter)


def swap_substrings(s: str, subs1: str, subs2: str) -> str:

    tmp1 = "X" * len(s)  # hacky but works
    tmp2 = "Y" * len(s)
    s = s.replace(subs1, tmp1)
    s = s.replace(subs2, tmp2)
    s = s.replace(tmp1, subs2)
    s = s.replace(tmp2, subs1)

    return s


class LMentryScorer:

    def __init__(self):
        self.prefixes = [""]
        self.prefix_kwargs = dict()
        self.suffixes = [""]
        self.suffix_kwargs = dict()

    @staticmethod
    def normalize_string(s: str):

        s = s.strip()
        # todo convert \W to its `[^]` equivalent before lowering (and after lowering reconvert)?
        #  be careful however, as I'm not sure \W is not sometimes \\W because of string vs
        #  regex string (r"") issues...
        s = s.lower()
        # convert consecutive spaces to a single space
        s = re.sub(r" +", " ", s)

        return s

    @staticmethod
    def normalize_prediction(prediction: str, truncate_prediction: bool = False):
        prediction = LMentryScorer.normalize_string(prediction)
        prediction = prediction.replace("_", " ")  # this is an easy hack to not have \w include "_"

        if truncate_prediction:
            prediction = prediction.split("\n")[0]
            prediction = prediction.split(".")[0]

        return prediction

    @staticmethod
    def get_shared_patterns(target):

        patterns = [
            rf"{target} is the answer",
            rf"The answer is {target}",
            rf"The correct answer is {target}",
            rf"Answer: {target}"
        ]
        return [p.format(target=target) for p in patterns]

    def get_patterns(self, base_patterns):

        # the x-fixes are the outer loop in order to yield all the non-x-fixed patterns first (thus
        #  saving time as they are expected to be much more common)
        for prefix, suffix in itertools.product(self.prefixes, self.suffixes):
            prefix = prefix.format(**self.prefix_kwargs)
            suffix = suffix.format(**self.suffix_kwargs)
            for base_pattern in base_patterns:
                yield prefix + base_pattern + suffix

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        raise NotImplementedError

    def score_predictions(self,
                          predictions_path: Path,
                          task_data_path: Path,
                          output_path: Path,
                          truncate_predictions: bool = False,
                          log_file_locations: bool = True
                          ):

        with open(task_data_path) as f_task_data:
            task_data = json.load(f_task_data)
        examples = task_data["examples"]

        # load the predictions without the metadata
        if log_file_locations:
            logging.info(f"loading predictions from {predictions_path}")
        with open(predictions_path) as f_predictions:
            predictions = json.load(f_predictions)

        # score predictions
        for id_ in predictions:
            example = examples[id_]
            prediction_entry = predictions[id_]
            prediction = prediction_entry["prediction"]
            score, certainty = self.score_prediction(prediction, example, truncate_predictions)
            prediction_entry["score"] = score
            prediction_entry["certainty"] = certainty

        # save the scored predictions
        with open(output_path, "w") as f_scored_predictions:
            json.dump(predictions, f_scored_predictions, indent=2)
        if log_file_locations:
            logging.info(f"saved scored predictions at {output_path}")

    def _simple_scorer(self, prediction, answer, allow_answer_pattern_repetitions=True) -> tuple[int, int]:

        score, certainty = None, None

        # return certain 0 if the prediction contains no alphanumeric character
        if not re.search(r"[^\W_]", prediction):  # we don't consider `_` to be alphanumeric
            return 0, 1

        if re.match(rf"{answer}\.?$", prediction, flags=re.IGNORECASE):
            score = 1
            certainty = 1

        if allow_answer_pattern_repetitions:
            alphanumeric_pattern = r"\b[a-z\d]+\b"
            all_alphanumeric_words = re.findall(alphanumeric_pattern, prediction)
            if all([re.match(answer + "$", word) for word in all_alphanumeric_words]):
                score = 1
                certainty = 1

        return score, certainty

    def certainty_scorer(self, prediction, base_patterns):
        score, certainty = 0, 0

        for i, pattern in enumerate(self.get_patterns(base_patterns)):
            pattern = self.normalize_string(pattern)
            if re.match(pattern + r"\.?$", prediction):
                score = 1
                certainty = 1
                break
            elif re.search(pattern, prediction):
                score = 1

        return score, certainty

    @staticmethod
    def display_prediction_patterns(task, model_name, show_only_uncertain_scores: bool = True):
        # load the predictions
        predictions_path = task.predictions_dir.joinpath(model_name).with_suffix(".json")
        with open(predictions_path) as f_predictions:
            predictions = json.load(f_predictions)

        prediction_patterns = []

        # load examples
        with open(task.default_data_path) as f_task_data:
            task_data = json.load(f_task_data)
        examples = task_data["examples"]

        for id_, prediction_entry in predictions.items():
            # take the prediction's pattern into account only if the prediction isn't scored yet
            if show_only_uncertain_scores and prediction_entry.get("certainty", 0) == 1:
                continue

            prediction = prediction_entry["prediction"]
            prediction = prediction.strip()
            for parameter_name in task.parameter_names:  # currently doesn't support parameters that are sequences
                argument = str(examples[id_]["metadata"][parameter_name])
                prediction = re.sub(f"\\b{argument}\\b", parameter_name.upper(), prediction,
                                    flags=re.IGNORECASE)
            prediction_patterns.append(prediction)

        prediction_pattern_counts = Counter(prediction_patterns)

        for pattern, count in prediction_pattern_counts.most_common():
            print(f"{count:<3}  {pattern}")
