import re

from lmentry.scorers.scorer import LMentryScorer, the_word_regex, the_letter_regex


class EndsWithLetterScorer(LMentryScorer):
    """This class was created by simply mirroring `StartsWithLetterScorer`
    """
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        word_ = r"(word|answer)"
        possible = r"possible( such)?"
        a = r"(a|one)"

        base_patterns = [
            rf"{a} {possible} {word_} is {word}",
            rf"{word} is {a} {possible} {word_}",
            rf"{word} ends with {letter}",
            rf"{word} is {a} word that ends with {letter}",
            rf"{word} is {a} word ending with {letter}",
            rf"A word ending with {letter} is {word}",
            rf"{word} is {a} word whose last letter is {letter}"
            rf"How about {word}",
            rf"May I suggest {word}",
        ]
        return base_patterns + self.get_shared_patterns(target=word)

    def negative_scorer(self, prediction, letter):
        score, certainty = None, None

        prediction_words = re.findall(r"\b[a-z]+\b", prediction)

        if all([word[-1] != letter for word in prediction_words]):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        score, certainty = self.negative_scorer(prediction, letter)
        if score is not None:
            return score, certainty

        before_the_letter = r"\w*" if letter in {"a", "i"} else r"\w+"

        word = rf"{before_the_letter}{letter}"
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
