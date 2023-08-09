import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from lmentry.constants import paper_models, hf_models, hf_11b_models


class ModelManager:
  def __init__(self, model_name: str):
    self.model_name = model_name
    if model_name in paper_models.keys():
      self.config = paper_models[model_name]
      self.type = "paper"
    elif model_name in hf_models.keys():
      self.config = hf_models[model_name]
      self.type = "hf"
    elif Path(model_name).is_dir():
      model_root = Path(model_name)
      # TODO(vvchernov): get from json?
      self.model_name = model_name
      self.config ={"artifact_path": model_name}
      self.type = "mlc"
    else:
      raise ValueError(f"Model name {model_name} is not in the list and not the path to mlc-llm model")

    self.short_name = self.config.get("short_name", self.model_name)
    self.paper_name = self.config.get("paper_name", self.model_name)
    self.predictor_name = self.config.get("predictor_name", self.model_name)

    self.tokenizer = None
    if self.type == "paper":
      self.tokenizer = AutoTokenizer.from_pretrained(self.predictor_name)
    elif self.type == "hf":
      self.tokenizer = AutoTokenizer.from_pretrained(self.predictor_name, padding_side='left')
    elif self.type == "mlc":
      self.tokenizer = AutoTokenizer.from_pretrained(
        Path(self.config["artifact_path"]).joinpath("params"), trust_remote_code=True
      )
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      if self.model_name.startswith("dolly-"):
          # 50277 means "### End"
          self.tokenizer.eos_token_id = 50277

    logging.info(f"loading model {self.predictor_name}")
    if self.type == "paper":
      self.model = AutoModelForSeq2SeqLM.from_pretrained(self.predictor_name)
    elif self.type == "hf":
      self.model = AutoModelForCausalLM.from_pretrained(self.predictor_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    elif self.type == "mlc":
      from lmentry.relax_model_wrapper import get_relax_model
      self.model = get_relax_model(self.config, self.tokenizer.eos_token_id)
    logging.info(f"finished loading model {self.predictor_name}")

    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

  def get_tokenizer(self):
    return self.tokenizer

  def to_device(self):
    if self.model_name in hf_11b_models:  # 11B models have to be parallelized
      self.model.parallelize()
    else:
      self.model.to(self.device)
