import json
import logging
from pathlib import Path

from lmentry.constants import TASKS_DATA_DIR
from tasks.lmentry.all_words_from_category import AllWordsFromCategory
from tasks.lmentry.all_words_from_category_0_distractors import AllWordsFromCategory0Distractors
from tasks.lmentry.all_words_from_category_1_distractors import AllWordsFromCategory1Distractors
from tasks.lmentry.all_words_from_category_2_distractors import AllWordsFromCategory2Distractors
from tasks.lmentry.any_words_from_category import AnyWordsFromCategory
from tasks.lmentry.any_words_from_category_3_distractors import AnyWordsFromCategory3Distractors
from tasks.lmentry.any_words_from_category_4_distractors import AnyWordsFromCategory4Distractors
from tasks.lmentry.any_words_from_category_5_distractors import AnyWordsFromCategory5Distractors
from tasks.lmentry.bigger_number import BiggerNumber
from tasks.lmentry.ends_with_letter import EndsWithLetter
from tasks.lmentry.ends_with_word import EndsWithWord
from tasks.lmentry.first_alphabetically import FirstAlphabetically
from tasks.lmentry.first_alphabetically_consecutive_first_letter import FirstAlphabeticallyConsecutiveFirstLetter
from tasks.lmentry.first_alphabetically_different_first_letter import FirstAlphabeticallyDifferentFirstLetter
from tasks.lmentry.first_alphabetically_far_first_letter import FirstAlphabeticallyFarFirstLetter
from tasks.lmentry.first_alphabetically_same_first_letter import FirstAlphabeticallySameFirstLetter
from tasks.lmentry.first_letter import FirstLetter
from tasks.lmentry.first_word import FirstWord
from tasks.lmentry.homophones import Homophones
from tasks.lmentry.last_letter import LastLetter
from tasks.lmentry.last_word import LastWord
from tasks.lmentry.least_associated_word import LeastAssociatedWord
from tasks.lmentry.less_letters import LessLetters
from tasks.lmentry.less_letters_length_diff_3plus import LessLettersLengthDiff3plus
from tasks.lmentry.less_letters_length_diff_1 import LessLettersLengthDiff1
from tasks.lmentry.more_letters import MoreLetters
from tasks.lmentry.more_letters_length_diff_3plus import MoreLettersLengthDiff3plus
from tasks.lmentry.more_letters_length_diff_1 import MoreLettersLengthDiff1
from tasks.lmentry.most_associated_word import MostAssociatedWord
from tasks.lmentry.rhyming_word import RhymingWord
from tasks.lmentry.rhyming_word_orthographically_different import RhymingWordOrthographicallyDifferent
from tasks.lmentry.rhyming_word_orthographically_similar import RhymingWordOrthographicallySimilar
from tasks.lmentry.sentence_containing import SentenceContaining
from tasks.lmentry.sentence_not_containing import SentenceNotContaining
from tasks.lmentry.smaller_number import SmallerNumber
from tasks.lmentry.starts_with_letter import StartsWithLetter
from tasks.lmentry.starts_with_word import StartsWithWord
from tasks.task import LMentryTask
from tasks.lmentry.word_after import WordAfter
from tasks.lmentry.word_before import WordBefore
from tasks.lmentry.word_containing import WordContaining
from tasks.lmentry.word_not_containing import WordNotContaining

core_tasks = {
    "sentence_containing": SentenceContaining,
    "sentence_not_containing": SentenceNotContaining,
    "word_containing": WordContaining,
    "word_not_containing": WordNotContaining,
    "most_associated_word": MostAssociatedWord,
    "least_associated_word": LeastAssociatedWord,
    "any_words_from_category": AnyWordsFromCategory,
    "all_words_from_category": AllWordsFromCategory,
    "first_alphabetically": FirstAlphabetically,
    "more_letters": MoreLetters,
    "less_letters": LessLetters,
    "bigger_number": BiggerNumber,
    "smaller_number": SmallerNumber,
    "rhyming_word": RhymingWord,
    "homophones": Homophones,
    "word_after": WordAfter,
    "word_before": WordBefore,
    "starts_with_word": StartsWithWord,
    "ends_with_word": EndsWithWord,
    "starts_with_letter": StartsWithLetter,
    "ends_with_letter": EndsWithLetter,
    "first_word": FirstWord,
    "last_word": LastWord,
    "first_letter": FirstLetter,
    "last_letter": LastLetter,
}

analysis_tasks = {
    "first_alphabetically_far_first_letter": FirstAlphabeticallyFarFirstLetter,
    "first_alphabetically_different_first_letter": FirstAlphabeticallyDifferentFirstLetter,
    "first_alphabetically_consecutive_first_letter": FirstAlphabeticallyConsecutiveFirstLetter,
    "first_alphabetically_same_first_letter": FirstAlphabeticallySameFirstLetter,
    "any_words_from_category_5_distractors": AnyWordsFromCategory5Distractors,
    "any_words_from_category_4_distractors": AnyWordsFromCategory4Distractors,
    "any_words_from_category_3_distractors": AnyWordsFromCategory3Distractors,
    "all_words_from_category_0_distractors": AllWordsFromCategory0Distractors,
    "all_words_from_category_1_distractors": AllWordsFromCategory1Distractors,
    "all_words_from_category_2_distractors": AllWordsFromCategory2Distractors,
    "rhyming_word_orthographically_similar": RhymingWordOrthographicallySimilar,
    "rhyming_word_orthographically_different": RhymingWordOrthographicallyDifferent,
    "more_letters_length_diff_3plus": MoreLettersLengthDiff3plus,
    "more_letters_length_diff_1": MoreLettersLengthDiff1,
    "less_letters_length_diff_3plus": LessLettersLengthDiff3plus,
    "less_letters_length_diff_1": LessLettersLengthDiff1,
}

all_tasks = core_tasks | analysis_tasks

# It is part from core tasks
sensetive_7b_model_tasks = {
    "bigger_number": BiggerNumber,
    "smaller_number": SmallerNumber,
    "first_alphabetically": FirstAlphabetically,
    "first_letter": FirstLetter,
    "most_associated_word": MostAssociatedWord,
}


def create_task_data(task_name: str):
  task: LMentryTask = all_tasks[task_name]()
  logging.info(f"creating data for task \"{task_name}\"")
  task.create_data()


def create_all_task_data():
  for task_name in all_tasks:
    create_task_data(task_name)


def count_examples(tasks_to_count: list[str] = None, tasks_data_dir: Path = None):
  tasks_data_dir = tasks_data_dir or TASKS_DATA_DIR

  if tasks_to_count is None:
    tasks_to_count = all_tasks

  n_examples = 0
  for task_name in tasks_to_count:
    task_data_path = tasks_data_dir.joinpath(f"{task_name}.json")
    with open(task_data_path) as f:
      task_data = json.load(f)
      n_examples += len(task_data["examples"])

  print(f"#examples in all tasks combined: {n_examples}")
  return n_examples
