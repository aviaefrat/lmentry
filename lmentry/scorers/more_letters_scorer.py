import re

from lmentry.scorers.scorer import (
    LMentryScorer, swap_substrings, the_word_regex, standardized_number_regex
)


class MoreLettersScorer(LMentryScorer):

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "Of the words {word1} and {word2}, ",
            "From the words {word1} and {word2}, ",
        ])

    def get_base_patterns(self, answer, distractor, diff):

        base_patterns = [
            rf"{answer} has more letters",
            rf"{answer} has more letters than {distractor}",
            rf"{answer} has {diff} more letter(s)?",
            rf"{answer} has {diff} more letter(s)? than {distractor}",
            rf"The word that has more letters is {answer}",
            rf"{answer} is the word that has more letters",
            rf"{answer} is longer",
            rf"{answer} is longer than {distractor}",
            rf"{answer} is longer than {distractor} by {diff} letter(s)?",
            rf"The longer word is {answer}",
            rf"The word that is longer is {answer}",
            rf"{answer} is the word that is longer",
        ]
        # replace more with less or fewer and swap answer and distractor
        less = r"(less|fewer)"
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("more", less)
            for s in base_patterns
            if "more" in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace longer with shorter and swap answer and distractor
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("longer", "shorter")
            for s in base_patterns
            if "longer" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None

        negative_patterns = [
            distractor,
            rf"{distractor} has more letters",
            rf"{distractor} has more letters than {answer}",
            rf"{distractor} has {diff} more letter(s)?",
            rf"{distractor} has {diff} more letter(s)? than {answer}",
            rf"The word that has more letters is {distractor}",
            rf"{distractor} is the word that has more letters",
            rf"{distractor} is longer",
            rf"{distractor} is longer by {diff} letter(s)?",
            rf"{distractor} is longer than {answer}",
            rf"{distractor} is longer than {answer} by {diff} letter(s)?",
            rf"The longer word is {distractor}",
            rf"The word that is longer is {distractor}",
            rf"{distractor} is the word that is longer",
        ]
        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        # the model correctly picks the longer word, but wrongly states the difference in letters:
        number_regex = r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|zero|a single)"
        wrong_diff_patterns = [
            rf"{answer} has {number_regex} more letter(s)?",
            rf"{answer} has {number_regex} more letter(s)? than {distractor}",
            rf"{answer} is longer by {number_regex} letter(s)?",
            rf"{answer} is longer than {distractor} by {number_regex} letter(s)?",
        ]
        for wrong_diff_pattern in wrong_diff_patterns:
            if res := re.match(wrong_diff_pattern, prediction):
                predicted_diff = res.group("number")
                if not re.match(diff, predicted_diff):
                    score = 0
                    certainty = 1
                    break

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]

        distractor = metadata["distractor"]
        diff = len(answer) - len(distractor)
        word1 = metadata["word1"]
        word2 = metadata["word2"]

        answer = the_word_regex(answer)
        distractor = the_word_regex(distractor)
        diff = standardized_number_regex(diff)
        word1 = the_word_regex(word1)
        word2 = the_word_regex(word2)

        score, certainty = self.negative_scorer(prediction, answer, distractor, diff)
        if score is not None:
            return score, certainty

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, distractor, diff)
        self.prefix_kwargs = {"word1": word1, "word2": word2}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
