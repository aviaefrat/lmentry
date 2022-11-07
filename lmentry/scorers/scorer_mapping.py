from lmentry.scorers.all_words_from_category_scorer import AllWordsFromCategoryScorer
from lmentry.scorers.any_words_from_category_scorer import AnyWordsFromCategoryScorer
from lmentry.scorers.bigger_number_scorer import BiggerNumberScorer
from lmentry.scorers.ends_with_letter_scorer import EndsWithLetterScorer
from lmentry.scorers.ends_with_word_scorer import EndsWithWordScorer
from lmentry.scorers.first_alphabetically_scorer import FirstAlphabeticallyScorer
from lmentry.scorers.first_letter_scorer import FirstLetterScorer
from lmentry.scorers.first_word_scorer import FirstWordScorer
from lmentry.scorers.homophone_scorer import HomophoneScorer
from lmentry.scorers.last_letter_scorer import LastLetterScorer
from lmentry.scorers.last_word_scorer import LastWordScorer
from lmentry.scorers.least_associated_word_scorer import LeastAssociatedWordScorer
from lmentry.scorers.less_letters_scorer import LessLettersScorer
from lmentry.scorers.more_letters_scorer import MoreLettersScorer
from lmentry.scorers.most_associated_word_scorer import MostAssociatedWordScorer
from lmentry.scorers.rhyming_word_scorer import RhymingWordScorer
from lmentry.scorers.sentence_containing_scorer import SentenceContainingScorer
from lmentry.scorers.sentence_not_containing_scorer import SentenceNotContainingScorer
from lmentry.scorers.smaller_number_scorer import SmallerNumberScorer
from lmentry.scorers.starts_with_letter_scorer import StartsWithLetterScorer
from lmentry.scorers.starts_with_word_scorer import StartsWithWordScorer
from lmentry.scorers.word_after_scorer import WordAfterScorer
from lmentry.scorers.word_before_scorer import WordBeforeScorer
from lmentry.scorers.word_containing_scorer import WordContainingScorer
from lmentry.scorers.word_not_containing_scorer import WordNotContainingScorer

task_name_to_scorer = {
    "starts_with_word": StartsWithWordScorer,
    "ends_with_word": EndsWithWordScorer,
    "starts_with_letter": StartsWithLetterScorer,
    "ends_with_letter": EndsWithLetterScorer,
    "most_associated_word": MostAssociatedWordScorer,
    "least_associated_word": LeastAssociatedWordScorer,
    "any_words_from_category": AnyWordsFromCategoryScorer,
    "all_words_from_category": AllWordsFromCategoryScorer,
    "sentence_containing": SentenceContainingScorer,
    "sentence_not_containing": SentenceNotContainingScorer,
    "word_containing": WordContainingScorer,
    "word_not_containing": WordNotContainingScorer,
    "rhyming_word": RhymingWordScorer,
    "homophones": HomophoneScorer,
    "first_word": FirstWordScorer,
    "last_word": LastWordScorer,
    "first_letter": FirstLetterScorer,
    "last_letter": LastLetterScorer,
    "word_after": WordAfterScorer,
    "word_before": WordBeforeScorer,
    "more_letters": MoreLettersScorer,
    "less_letters": LessLettersScorer,
    "first_alphabetically": FirstAlphabeticallyScorer,
    "bigger_number": BiggerNumberScorer,
    "smaller_number": SmallerNumberScorer
}