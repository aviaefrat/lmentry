import re

from lmentry.scorers.scorer import LMentryScorer, the_sentence_regex


class SentenceContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()

        allowed_sentence = rf"((.*\b{word}\b.+)|(.+\b{word}\b.*))"
        allowed_sentence = the_sentence_regex(allowed_sentence)

        # I'm not using the LMentryScorer interface for this task, as I found it hard to adhere to.
        #  Instead, I wrote a naive scorer.
        if re.search(allowed_sentence, prediction):
            score = 1
            certainty = 1
        else:
            score = 0
            certainty = 1

        return score, certainty