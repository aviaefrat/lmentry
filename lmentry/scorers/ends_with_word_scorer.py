import re

from lmentry.scorers.scorer import LMentryScorer, the_word_regex


class EndsWithWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        a_ = r"(a|one)"
        last_ = r"(last|final)"

        # we don't want an empty prefix here
        self.prefixes = [
            r"The sentence( is)?",
            r"{a_} sentence that ends with {word} ( is)?",
            r"{a_} sentence ending with {word}( is)?",
            r"{a_} sentence whose {last_} word is {word}( is)?",
        ]

        self.suffixes = [
            r"$"
        ]

        def _replace_arguments(xfixes: list[str]):
            for i in range(len(xfixes)):
                xfixes[i] = xfixes[i].format(
                                             a_=a_,
                                             last_=last_,
                                             word="{word}")  # `word` will be replaced later
        _replace_arguments(self.prefixes)
        _replace_arguments(self.suffixes)

    def get_base_patterns(self, word):

        # we want a word to contain at least two letters, with the exception of "a" and "i"
        valid_word = r"(a|i|\w\w+)"
        # we use `[^a-zA-Z0-9_]` instead of `\W` as we always lowercase the pattern in
        # `_certainty_scorer`.
        # we can't tell if the sentence words are actual English words, but as a sanity we demand at
        # least one additional "word" before the ending word
        allowed_sentence = rf".*{valid_word}[^a-zA-Z0-9_]+{word}[^a-zA-Z0-9_]*"

        base_patterns = [
            allowed_sentence
        ]
        return base_patterns + self.get_shared_patterns(target=allowed_sentence)
    
    def negative_scorer(self, prediction, word):
        score, certainty = None, None

        if not re.search(rf"\b{word}\b", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()
        score, certainty = self.negative_scorer(prediction, word)
        if score is not None:
            return score, certainty

        word = the_word_regex(word)

        base_patterns = self.get_base_patterns(word)

        # note that the second case of `_simple_scorer` is not relevant
        score, certainty = self._simple_scorer(prediction, base_patterns[0])
        if score:
            return score, certainty

        self.prefix_kwargs = {"word": word}
        self.suffix_kwargs = {"word": word}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
