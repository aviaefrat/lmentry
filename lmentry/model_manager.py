import logging

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from lmentry.constants import paper_models, hf_models
from lmentry.relax_model_wrapper import RelaxModelWrapper


class ModelManager:
  def __init__(self, model_name: str):
    self.model_name = model_name
    if model_name in paper_models.keys():
      self.config = paper_models[model_name]
      self.type = "paper"
    elif model_name in hf_models.keys():
      self.config = hf_models[model_name]
      self.type = "hf"

    self.short_name = self.config.get("short_name", model_name)
    self.paper_name = self.config.get("paper_name", model_name)
    self.predictor_name = self.config.get("predictor_name", model_name)

    logging.info(f"loading model {self.predictor_name}")
    if self.type == "paper":
      self.model = AutoModelForSeq2SeqLM.from_pretrained(self.predictor_name)
    elif self.type == "hf":
      self.model = AutoModelForCausalLM.from_pretrained(self.predictor_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    logging.info(f"finished loading model {self.predictor_name}")

  def get_tokenizer(self):
    return AutoTokenizer.from_pretrained(self.predictor_name, padding_side='left')
