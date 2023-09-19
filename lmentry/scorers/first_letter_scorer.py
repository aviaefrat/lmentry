import re

from lmentry.scorers.scorer import LMentryScorer, the_word_regex, the_letter_regex


class FirstLetterScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, word):

        of = "(of|in)"
        starts = "(starts|begins)"

        base_patterns = [
            rf"The first letter is {answer}",
            rf"The first letter {of} {word} is {answer}",
            rf"{answer} is the first letter {of} {word}",
            rf"{word} {starts} with {answer}",
            rf"The letter that {word} {starts} with is {answer}",
            rf"The letter {answer} {starts} with the word {word}",
            rf"{answer} is the starting letter {of} {word}",
            rf"{word}: {answer}",
            rf"First letter: {answer}",
        ]
        return base_patterns + self.get_shared_patterns(target=answer)

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]
        word = metadata["word"]

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        answer = the_letter_regex(answer)

        score, certainty = self._simple_scorer(prediction, answer, False)
        if score:
            return score, certainty

        word = the_word_regex(word)

        base_patterns = self.get_base_patterns(answer, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
