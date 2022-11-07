import re

from lmentry.scorers.scorer import LMentryScorer, the_word_regex, the_letter_regex


class StartsWithLetterScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        word_ = r"(word|answer)"
        possible = r"possible( such)?"
        starts = r"(starts|begins)"
        starting = r"(starting|beginning)"
        a = r"(a|one)"

        base_patterns = [
            rf"{a} {possible} {word_} is {word}",
            rf"{word} is {a} {possible} {word_}",
            rf"{word} {starts} with {letter}",
            rf"{word} is {a} word that {starts} with {letter}",
            rf"{word} is {a} word {starting} with {letter}",
            rf"A word starting with {letter} is {word}",
            rf"{word} is {a} word whose first letter is {letter}",
            rf"{letter} is for {word}",
            rf"How about {word}",
            rf"May I suggest {word}",
        ]
        return base_patterns + self.get_shared_patterns(target=word)

    def negative_scorer(self, prediction, letter):
        score, certainty = None, None

        prediction_words = re.findall(r"\b[a-z]+\b", prediction)

        if all([word[0] != letter for word in prediction_words]):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        score, certainty = self.negative_scorer(prediction, letter)
        if score is not None:
            return score, certainty

        after_the_letter = r"\w*" if letter in {"a", "i"} else r"\w+"

        word = rf"{letter}-?{after_the_letter}"  # the "-" is for cases like e-mail and x-ray
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
