from lmentry.scorers.scorer import LMentryScorer, the_number_regex, the_word_regex, the_words_regex


class HFTaskScorer(LMentryScorer):
  def __init__(self):
    super().__init__()

  def score_prediction(self, prediction, example, truncate_prediction: bool = False):
    prediction = self.normalize_prediction(prediction, truncate_prediction)

    metadata = example["metadata"]
    answer = metadata["answer"]

    # TODO(vchernov): transfer to task create?
    answer = " ".join(answer.split())

    print("Prediction:", prediction)
    print("Answer:", answer)
    score, certainty = self.negative_scorer(prediction, answer)
    if score is not None:
      return score, certainty
    print("Pass negative scorer")

    if answer.isnumeric():
      answer = the_number_regex(answer)
    elif " " in answer:
      answer = the_words_regex(answer)
    else:
      answer = the_word_regex(answer)

    score, certainty = self._simple_scorer(prediction, answer)
    if score:
      return score, certainty
    print("Pass simple scorer")

    base_patterns = self.get_shared_patterns(target=answer)

    score, certainty = self.certainty_scorer(prediction, base_patterns)
    return score, certainty
