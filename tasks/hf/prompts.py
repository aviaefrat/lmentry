from tasks.hf.utils import pattern_match


def load_prompt_list(use_prompt: str, dataset_name=None, subset_name=None, **kwargs):
  from promptsource.templates import DatasetTemplates

  if subset_name is None:
    prompts = DatasetTemplates(dataset_name=dataset_name)
  else:
    prompts = DatasetTemplates(dataset_name=dataset_name, subset_name=subset_name)

  category_name, *prompt_name = use_prompt.split(":")
  prompt_list = pattern_match(prompt_name, prompts.all_template_names)
  return [":".join([category_name, prompt]) for prompt in prompt_list]
