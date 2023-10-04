from lmentry.scorers.scorer import LMentryScorer, the_word_regex


class CapitalScorer(LMentryScorer):
  def __init__(self):
    super().__init__()

  def get_base_patterns(self, country, answer):
    base_patterns = [
        rf"{answer} is the capital of {country}",
        rf"The capital of {country} is {answer}",
    ]

    return base_patterns + self.get_shared_patterns(target=answer)

  def score_prediction(self, prediction, example, truncate_prediction: bool = False):
    prediction = self.normalize_prediction(prediction, truncate_prediction)

    metadata = example["metadata"]
    answer = metadata["answer"]

    score, certainty = self.negative_scorer(prediction, answer)
    if score is not None:
        return score, certainty

    country = metadata["country"]

    # TODO(vvchernov): check many words and use the_words_regex
    answer = the_word_regex(answer)
    country = the_word_regex(country)

    score, certainty = self._simple_scorer(prediction, answer)
    if score:
        return score, certainty

    base_patterns = self.get_base_patterns(country, answer)

    score, certainty = self.certainty_scorer(prediction, base_patterns)
    return score, certainty


class CapitalRuScorer(LMentryScorer):
  def __init__(self):
    super().__init__()

  def get_base_patterns(self, country, answer):
    base_patterns = [
        rf"{answer} столица {country}",
        rf"Столица {country} {answer}",
    ]

    return base_patterns + self.get_shared_patterns_ru(target=answer)

  def score_prediction(self, prediction, example, truncate_prediction: bool = False):
    prediction = self.normalize_prediction(prediction, truncate_prediction)

    metadata = example["metadata"]
    answer = metadata["answer"]

    score, certainty = self.negative_scorer(prediction, answer)
    if score is not None:
        return score, certainty

    country = metadata["country"]

    # TODO(vvchernov): check many words and use the_words_regex
    answer = the_word_regex(answer)
    country = the_word_regex(country)

    score, certainty = self._simple_scorer(prediction, answer)
    if score:
        return score, certainty

    base_patterns = self.get_base_patterns(country, answer)

    score, certainty = self.certainty_scorer(prediction, base_patterns)
    return score, certainty
