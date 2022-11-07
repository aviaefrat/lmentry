import re

from lmentry.scorers.scorer import (
    LMentryScorer, swap_substrings, the_word_regex, standardized_number_regex
)


class LessLettersScorer(LMentryScorer):
    """This class was created by simply mirroring `MoreLettersScorer`
    """
    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "Of the words {word1} and {word2}, ",
            "From the words {word1} and {word2}, ",
        ])

    def get_base_patterns(self, answer, distractor, diff):

        less = r"(less|fewer)"
        base_patterns = [
            rf"{answer} has {less} letters",
            rf"{answer} has {less} letters than {distractor}",
            rf"{answer} has {diff} {less} letter(s)?",
            rf"{answer} has {diff} {less} letter(s)? than {distractor}",
            rf"The word that has {less} letters is {answer}",
            rf"{answer} is the word that has {less} letters",
            rf"{answer} is shorter",
            rf"{answer} is shorter than {distractor}",
            rf"{answer} is shorter than {distractor} by {diff} letter(s)?",
            rf"The shorter word is {answer}",
            rf"The word that is shorter is {answer}",
            rf"{answer} is the word that is shorter",
        ]
        # replace less with more and swap answer and distractor
        less = r"(less|fewer)"
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace(less, "more")
            for s in base_patterns
            if less in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace shorter with longer and swap answer and distractor
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("shorter", "longer")
            for s in base_patterns
            if "shorter" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)
    
    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None

        less = r"(less|fewer)"
        negative_patterns = [
            distractor,
            rf"{distractor} has {less} letters",
            rf"{distractor} has {less} letters than {answer}",
            rf"{distractor} has {diff} {less} letter(s)?",
            rf"{distractor} has {diff} {less} letter(s)? than {answer}",
            rf"The word that has {less} letters is {distractor}",
            rf"{distractor} is the word that has {less} letters",
            rf"{distractor} is shorter",
            rf"{distractor} is shorter by {diff} letter(s)?",
            rf"{distractor} is shorter than {answer}",
            rf"{distractor} is shorter than {answer} by {diff} letter(s)?",
            rf"The shorter word is {distractor}",
            rf"The word that is shorter is {distractor}",
            rf"{distractor} is the word that is shorter",
        ]
        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        # the model correctly picks the shorter word, but wrongly states the difference in letters:
        number_regex = r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|zero|a single)"
        wrong_diff_patterns = [
            rf"{answer} has {number_regex} {less} letter(s)?",
            rf"{answer} has {number_regex} {less} letter(s)? than {distractor}",
            rf"{answer} is shorter by {number_regex} letter(s)?",
            rf"{answer} is shorter than {distractor} by {number_regex} letter(s)?",
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
        diff = len(distractor) - len(answer)
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
