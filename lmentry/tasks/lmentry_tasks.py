import json
import logging
from pathlib import Path
from typing import List, Union

from lmentry.constants import TASKS_DATA_DIR
from lmentry.tasks.all_words_from_category import AllWordsFromCategory
from lmentry.tasks.all_words_from_category_0_distractors import AllWordsFromCategory0Distractors
from lmentry.tasks.all_words_from_category_1_distractors import AllWordsFromCategory1Distractors
from lmentry.tasks.all_words_from_category_2_distractors import AllWordsFromCategory2Distractors
from lmentry.tasks.any_words_from_category import AnyWordsFromCategory
from lmentry.tasks.any_words_from_category_3_distractors import AnyWordsFromCategory3Distractors
from lmentry.tasks.any_words_from_category_4_distractors import AnyWordsFromCategory4Distractors
from lmentry.tasks.any_words_from_category_5_distractors import AnyWordsFromCategory5Distractors
from lmentry.tasks.bigger_number import BiggerNumber
from lmentry.tasks.ends_with_letter import EndsWithLetter
from lmentry.tasks.ends_with_word import EndsWithWord
from lmentry.tasks.first_alphabetically import FirstAlphabetically
from lmentry.tasks.first_alphabetically_consecutive_first_letter import FirstAlphabeticallyConsecutiveFirstLetter
from lmentry.tasks.first_alphabetically_different_first_letter import FirstAlphabeticallyDifferentFirstLetter
from lmentry.tasks.first_alphabetically_far_first_letter import FirstAlphabeticallyFarFirstLetter
from lmentry.tasks.first_alphabetically_same_first_letter import FirstAlphabeticallySameFirstLetter
from lmentry.tasks.first_letter import FirstLetter
from lmentry.tasks.first_word import FirstWord
from lmentry.tasks.homophones import Homophones
from lmentry.tasks.last_letter import LastLetter
from lmentry.tasks.last_word import LastWord
from lmentry.tasks.least_associated_word import LeastAssociatedWord
from lmentry.tasks.less_letters import LessLetters
from lmentry.tasks.less_letters_length_diff_3plus import LessLettersLengthDiff3plus
from lmentry.tasks.less_letters_length_diff_1 import LessLettersLengthDiff1
from lmentry.tasks.more_letters import MoreLetters
from lmentry.tasks.more_letters_length_diff_3plus import MoreLettersLengthDiff3plus
from lmentry.tasks.more_letters_length_diff_1 import MoreLettersLengthDiff1
from lmentry.tasks.most_associated_word import MostAssociatedWord
from lmentry.tasks.rhyming_word import RhymingWord
from lmentry.tasks.rhyming_word_orthographically_different import RhymingWordOrthographicallyDifferent
from lmentry.tasks.rhyming_word_orthographically_similar import RhymingWordOrthographicallySimilar
from lmentry.tasks.sentence_containing import SentenceContaining
from lmentry.tasks.sentence_not_containing import SentenceNotContaining
from lmentry.tasks.smaller_number import SmallerNumber
from lmentry.tasks.starts_with_letter import StartsWithLetter
from lmentry.tasks.starts_with_word import StartsWithWord
from lmentry.tasks.task import LMentryTask
from lmentry.tasks.word_after import WordAfter
from lmentry.tasks.word_before import WordBefore
from lmentry.tasks.word_containing import WordContaining
from lmentry.tasks.word_not_containing import WordNotContaining
from lmentry.tasks.simple_tasks import Capital

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

simple_tasks = {
    "capital" : Capital,
}

all_tasks = core_tasks | analysis_tasks | simple_tasks

# It is part from core tasks
sensetive_7b_model_tasks = {
    "bigger_number": BiggerNumber,
    "smaller_number": SmallerNumber,
    "first_alphabetically": FirstAlphabetically,
    "first_letter": FirstLetter,
    "most_associated_word": MostAssociatedWord,
}

tasks_list = {
    "all": all_tasks,
    "core": core_tasks,
    "analysis": analysis_tasks,
    "simple": simple_tasks,
    "7b": sensetive_7b_model_tasks,
}


def get_tasks_names(task_names: Union[List[str], None]=["all"]):
  if not task_names:
    return None
  ret_task_names = []
  for task_name in task_names:
    if task_name in tasks_list.keys():
      ret_task_names = ret_task_names + sorted(tasks_list[task_name].keys())
    elif task_name in all_tasks.keys():
      ret_task_names.append()
    else:
      raise ValueError(f"There is no task or task set with name {task_name}")
  return ret_task_names


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
