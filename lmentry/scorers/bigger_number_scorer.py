import re

from lmentry.scorers.scorer import LMentryScorer, swap_substrings, the_number_regex


class BiggerNumberScorer(LMentryScorer):

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "Of the numbers {n1} and {n2}, ",
            "From the numbers {n1} and {n2}, ",
            "Between the numbers {n1} and {n2}, ",
        ])

    def get_base_patterns(self, answer, distractor):

        bigger = r"(bigger|larger)"

        base_patterns = [
            rf"{answer} is {bigger}",
            rf"{answer} is {bigger} than {distractor}",
            rf"{answer} is the {bigger} number",
            rf"The {bigger} number is {answer}",
            rf"The {bigger} is {answer}",
            rf"{distractor}(\s|)<(\s|){answer}",
            rf"{answer}(\s|)>(\s|){distractor}",
            rf"The number that is {bigger} than {distractor} is {answer}.",
        ]
        # replace bigger with smaller and swap answer and distractor
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace(bigger, "smaller")
            for s in base_patterns
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        alphanumeric_pattern = r"\b[a-z\d]+\b"
        all_alphanumeric_words = re.findall(alphanumeric_pattern, prediction)

        if len(set(all_alphanumeric_words)) == 1:
            answer = str(answer)
            if all_alphanumeric_words[0] != answer:
                score = 0
                certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        n1 = metadata["n1"]
        n2 = metadata["n2"]
        answer = metadata["answer"]

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        distractor = metadata["distractor"]

        answer = the_number_regex(answer)
        distractor = the_number_regex(distractor)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, distractor)
        self.prefix_kwargs = {"n1": n1, "n2": n2}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
