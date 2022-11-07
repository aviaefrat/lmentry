import re

from lmentry.scorers.scorer import LMentryScorer, the_word_regex, swap_substrings


class RhymingWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        a = r"(a|the)"
        while_ = "(while|but|and)"
        doesnt = r"(does not|doesn't)"
        exact_patterns = [
            rf"{query} rhymes with {answer}( and not with {distractor})?",
            rf"{query} is {a} rhyme of {answer}( and not of {distractor})?",
            rf"{query} and {answer} rhyme",
            rf"{query} and {answer} are rhymes",
            rf"{query} and {answer} are words that rhyme",
            rf"{a} rhyme of {query} is {answer}( and not {distractor})?",
            rf"{a} word rhyming with {query} is {answer}",
            rf"{a} word that rhymes with {query} is {answer}",
            rf"{query} and {answer}( both)? rhyme( with each other)?, {while_} {distractor} {doesnt} rhyme with either( word)?"
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        more_exact_patterns_2 = [
            rf"{distractor} {doesnt} rhyme with {query}(\s+{answer} (rhymes|does rhyme) with {query})?",
            rf"{query} {doesnt} rhyme with {distractor}(\s+{query} (rhymes|does rhyme) with {answer})?",
            rf"{answer} rhymes with {query}\s+{distractor} {doesnt} rhyme with {query}",
        ]
        exact_patterns.extend(more_exact_patterns_2)

        return exact_patterns

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        predictions_words = re.findall(r"\b[a-z]+\b", prediction)
        if len(predictions_words) == 1 and predictions_words[0] != answer:
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]
        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        answer = the_word_regex(answer)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        query = metadata["query"]
        distractor = metadata["distractor"]
        query = the_word_regex(query)
        distractor = the_word_regex(distractor)
        exact_patterns = self.get_exact_patterns(answer, query, distractor)

        for exact_pattern in exact_patterns:
            score, certainty = self._simple_scorer(prediction, exact_pattern)
            if score:
                return score, certainty
        else:
            score = 0
            certainty = 0

        return score, certainty
