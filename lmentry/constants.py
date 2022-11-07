from pathlib import Path

lmentry_random_seed = 1407

ROOT_DIR = Path(__file__).parent.parent
TASKS_DATA_DIR = ROOT_DIR.joinpath("data")
PREDICTIONS_ROOT_DIR = ROOT_DIR.joinpath("predictions")
RESULTS_DIR = ROOT_DIR.joinpath("results")
VISUALIZATIONS_DIR = RESULTS_DIR.joinpath("visualizations")
SCORER_EVALUATION_DIR = RESULTS_DIR.joinpath("scorer_evaluation")
RESOURCES_DIR = ROOT_DIR.joinpath("resources")
LMENTRY_WORDS_PATH = RESOURCES_DIR.joinpath("lmentry_words.json")
RHYME_GROUPS_PATH = RESOURCES_DIR.joinpath("rhyme_groups.json")
HOMOPHONES_PATH = RESOURCES_DIR.joinpath("homophones.csv")
PHONETICALLY_UNAMBIGUOUS_WORDS_PATH = RESOURCES_DIR.joinpath("phonetically_unambiguous_words.json")


paper_models = {
    "text-davinci-002": {"short_name": "d-002", "paper_name": "TextDavinci002"},
    "text-davinci-001": {"short_name": "d-001", "paper_name": "\igpt{} 175B"},
    "text-curie-001": {"short_name": "c-001", "paper_name": "\igpt{} 6.7B"},
    "text-babbage-001": {"short_name": "b-001", "paper_name": "\igpt{} 1.3B"},
    "text-ada-001": {"short_name": "a-001", "paper_name": "\igpt{} 0.35B"},
    "davinci": {"short_name": "d-000", "paper_name": "GPT-3 175B"},
    "curie": {"short_name": "c-000", "paper_name": "GPT-3 6.7B"},
    "babbage": {"short_name": "b-000", "paper_name": "GPT-3 1.3B"},
    "ada": {"short_name": "a-000", "paper_name": "GPT-3 0.35B"},
    "bigscience-T0pp": {"short_name": "T0pp", "paper_name": "T0pp 11B", "predictor_name": "bigscience/T0pp"},
    "bigscience-T0p": {"short_name": "T0p", "paper_name": "T0p 11B", "predictor_name": "bigscience/T0p"},
    "bigscience-T0": {"short_name": "T0", "paper_name": "T0 11B", "predictor_name": "bigscience/T0"},
    "allenai-tk-instruct-11b-def-pos-neg-expl": {"short_name": "Tk-dpne", "paper_name": "Tk-dpne 11B", "predictor_name": "allenai/tk-instruct-11b-def-pos-neg-expl"},
    "allenai-tk-instruct-11b-def-pos": {"short_name": "Tk-dp", "paper_name": "Tk-dp 11B", "predictor_name": "allenai/tk-instruct-11b-def-pos"},
    "allenai-tk-instruct-11b-def": {"short_name": "Tk-d", "paper_name": "Tk-d 11B", "predictor_name": "allenai/tk-instruct-11b-def"},
    "google-t5-xxl-lm-adapt": {"short_name": "T5xxl", "paper_name": "T5-LM 11B", "predictor_name": "google/t5-xxl-lm-adapt"},
}

text_001_models = [
    "text-davinci-001",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001"
]
gpt3_vanilla_models = [
    "davinci",
    "curie",
    "babbage",
    "ada"
]
open_ai_instruction_models = ["text-davinci-002"] + text_001_models
t0_11b_models = [
    "bigscience-T0pp",
    "bigscience-T0p",
    "bigscience-T0"
]
tk_11b_models = [
    "allenai-tk-instruct-11b-def-pos-neg-expl",
    "allenai-tk-instruct-11b-def-pos",
    "allenai-tk-instruct-11b-def"
]

hf_11b_models = t0_11b_models + tk_11b_models


def get_short_model_name(model_name: str):
    return paper_models[model_name]["short_name"]


def get_predictor_model_name(model_name: str):
    return paper_models[model_name].get("predictor_name", model_name)


def get_paper_model_name(model_name: str):
    return paper_models[model_name]["paper_name"]


simple_answer_index_task_names = [
    "more_letters", "less_letters", "first_alphabetically",
    "rhyming_word", "homophones", "bigger_number", "smaller_number"
]

models_that_cant_stop_generating = {"davinci", "curie", "babbage", "ada",
                                    "google-t5-xxl-lm-adapt"}
