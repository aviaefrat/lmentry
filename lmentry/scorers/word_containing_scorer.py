from lmentry.scorers.scorer import LMentryScorer, the_word_regex, the_letter_regex


class WordContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        that = r"(that|which)"
        contains = r"(contains|uses|has)"

        base_patterns = [
            rf"{word} (is a word {that} )?{contains} {letter}",
        ]

        return base_patterns + self.get_shared_patterns(target=word)

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        word = (rf"\w*{letter}\w*"
                if letter in {"a", "i"}
                else rf"((\w+{letter}\w*)|(\w*{letter}\w+)|({letter}-\w+))"
                # the third option under `else` is for cases like "e-mail" and "x-ray"
                )
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)
        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
