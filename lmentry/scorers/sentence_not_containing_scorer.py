import re

from lmentry.scorers.scorer import LMentryScorer


class SentenceNotContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()

        # I wrote a custom naive scorer, as I found it hard to adhere to the LMentryScorer interface for this task

        if re.search(f'\\b{word}\\b', prediction):
            score = 0
            certainty = 1
        else:
            prediction_words = re.findall(r"\b[a-z]+\b", prediction)
            if len(prediction_words) == 0:
                score = 0
                certainty = 1
            elif len(prediction_words) == 1:  # we don't count one word as a sentence, but it can be argued otherwise
                score = 0
                certainty = 0
            else:  # the prediction is two words or more
                if word in prediction:  # the word is a substring of another word, e.g. cat->cats
                    score = 1
                    certainty = 0
                else:  # the word is not even a substring of the prediction
                    score = 1
                    certainty = 1

        return score, certainty
