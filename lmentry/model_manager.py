import logging
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from lmentry.constants import paper_models, hf_models, hf_11b_models


def get_type_config(
    model_name: str,
    device: str="cuda",
    use_vllm: bool=False,
    name_from_mlc_config: bool=False,
):
  type=""
  config={}
  if model_name in paper_models.keys():
    type = "paper"
    config = paper_models[model_name]
    config["model_name"] = model_name
  elif model_name in hf_models.keys():
    if use_vllm:
      type = "vllm"
      config = hf_models[model_name]
      config["model_name"] = model_name
    else:
      type = "hf"
      config = hf_models[model_name]
      config["model_name"] = model_name
  elif Path(model_name).is_dir():
    type = "mlc"
    model_root = Path(model_name)

    mlc_config_file = model_root.joinpath("params/mlc-chat-config.json")

    mlc_config = {}
    with open(mlc_config_file) as json_file:
      mlc_config = json.load(json_file)

    model_local_id = mlc_config["local_id"]
    if name_from_mlc_config:
      config_model_name = model_local_id
    else:
      config_model_name = str(model_root).split("/")[-1]
    config_model_name = config_model_name.replace(".", "-")

    config ={
      "model_name": config_model_name,
      "artifact_path": model_name,
      "mlc_model_name": model_local_id,
      "device": device,
      "temperature": mlc_config["temperature"],
      "top_p": mlc_config["top_p"],
    }
  else:
    raise ValueError(f"Model name {model_name} is not in the list and not the path to mlc-llm model")

  model_name = config["model_name"]
  config["short_name"] = config.get("short_name", model_name)
  config["paper_name"] = config.get("paper_name", model_name)
  config["predictor_name"] = config.get("predictor_name", model_name)

  return type, config


def get_short_model_names(model_names):
  short_model_names =[]
  for model_name in model_names:
    _, model_config = get_type_config(model_name)
    short_model_names.append(model_config["short_name"])

  return short_model_names


class ModelManager:
  def __init__(self, model_name: str, device: str="cuda", max_length: int=100, use_vllm: bool=False):
    self.max_length = max_length
    self.type, self.config = get_type_config(model_name, device, use_vllm)
    self.model_name = self.config["model_name"]

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
    elif self.type == "vllm":
      from lmentry.vllm_model_wrapper import VllmModelWrapper
      self.tokenizer = VllmModelWrapper.get_vllm_tokenizer(tokenizer_name=model_name, trust_remote_code=True)

    self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    self.model = None
    self.is_init = False

  def init_model(self):
    if not self.is_init:
      logging.info(f"Initializing model {self.predictor_name}")
      if self.type == "paper":
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.predictor_name)
      elif self.type == "hf":
        self.model = AutoModelForCausalLM.from_pretrained(self.predictor_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
      elif self.type == "mlc":
        from lmentry.relax_model_wrapper import get_relax_model
        self.model = get_relax_model(self.config, self.tokenizer.eos_token_id)
      elif self.type == "vllm":
        from lmentry.vllm_model_wrapper import VllmModelWrapper
        
        # TODO(vvchernov): recheck [updated]
        model_name = self.predictor_name
        self.model = VllmModelWrapper.get_vllm_model(model_name, self.config, self.max_length, model_name)
      logging.info(f"finished initializing model {self.predictor_name}")
      self.is_init = True

  def is_init(self):
    return self.is_init

  def get_tokenizer(self):
    return self.tokenizer

  def to_device(self):
    if self.type != "mlc" and self.type != "vllm":
      if self.is_init:
        if self.model_name in hf_11b_models:  # 11B models have to be parallelized
          self.model.parallelize()
        else:
          self.model.to(self.device)
      else:
        raise RuntimeError(f"Model {self.predictor_name} was not initialized!")
