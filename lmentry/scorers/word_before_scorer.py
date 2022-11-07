import re

from lmentry.scorers.scorer import LMentryScorer, the_sentence_regex, the_word_regex, swap_substrings


class WordBeforeScorer(LMentryScorer):
    """This class was created by simply mirroring `WordAfterScorer`
    """
    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "In {sentence} ",
        ])

        self.suffixes.extend([
            " (of|in) {sentence}",
        ])

    def get_base_patterns(self, answer, query, sentence):

        before = r"(immediately |right )?before"
        precedes = r"(immediately )?precedes"
        follows = r"(immediately )?follows"

        base_patterns = [
            rf"{answer} comes {before} {query}",
            rf"{answer} is the word that comes {before} {query}",
            rf"{answer} {precedes} {query}",
            rf"{answer} is the word that {precedes} {query}",
            rf"{answer} is followed by {query}",
            rf"The word that comes {before} {query} is {answer}",
            rf"The word that comes {before} {query} in {sentence} is {answer}",
            rf"The word that {precedes} {query} is {answer}",
            rf"The word that {follows} {answer} is {query}",
            rf"{answer} is succeeded by {query}",
            rf"{query} {follows} {answer}",
        ]

        # replace before with after and swap answer and query
        after = r"(immediately |right )?after"
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=query).replace(before, after)
            for s in base_patterns
            if before in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace precedes with succeeds and swap answer and query
        succeeds = r"(immediately )?succeeds"
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=query).replace(precedes, succeeds)
            for s in base_patterns
            if succeeds in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, sentence):
        score, certainty = None, None

        if not re.search(rf"\b{answer}\b", prediction):
            score = 0
            certainty = 1

        if re.sub(r'[^\w ]+', '', prediction) == sentence:
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]
        query = metadata["query"]
        sentence = metadata["sentence"]

        # in the word before/after tasks, the sentence, the answer, and the query are saved in their
        # original case as many sentence words are named entities. in scoring, we choose to ignore
        # the case.
        answer = answer.lower()
        query = query.lower()
        sentence = sentence.lower()

        score, certainty = self.negative_scorer(prediction, answer, sentence)
        if score is not None:
            return score, certainty

        answer = the_word_regex(answer)
        query = the_word_regex(query)
        sentence = the_sentence_regex(sentence)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, query, sentence)
        self.prefix_kwargs = {"sentence": sentence}
        self.suffix_kwargs = {"sentence": sentence}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
