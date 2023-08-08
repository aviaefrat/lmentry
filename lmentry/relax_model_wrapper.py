import argparse
from typing import Callable

import torch


def sample_top_p(probs, p):
  probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
  probs_sum = torch.cumsum(probs_sort, dim=-1)
  mask = probs_sum - probs_sort > p
  probs_sort[mask] = 0.0
  probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
  next_token = torch.multinomial(probs_sort, num_samples=1)
  next_token = torch.gather(probs_idx, -1, next_token)
  return next_token


class RelaxModelWrapper:
  def __init__(self,
               model: Callable,
               tokenizer,
               args: argparse.Namespace,
               temperature: float = 1.1,
               top_p: float = 0.7,
  ):
    self.model = model
    self.stop_tokens = [tokenizer.eos_token_id]
    self.tokenizer = tokenizer
    self.args = args

    self.temperature = temperature
    self.top_p = top_p

  def generate(
    self,
    tokens,
    max_length: int,
  ):
    prompt_len = tokens.shape[1]
    total_len = max_length + prompt_len
    start_pos = prompt_len
    for cur_pos in range(start_pos, total_len):
      if cur_pos == start_pos:
        logits = self.model(tokens[:, :cur_pos])
      else:
        logits = self.model(tokens[:, cur_pos - 1 : cur_pos])
      logits = logits[:, -1, :].to(torch.float64)
      np_logits = logits.detach().cpu().numpy().astype("float64")
      if self.args.debug_dump:
        print(
          f"logits: min = {np_logits.min()}, max = {np_logits.max()}, "
          f"mean = {np_logits.mean()}, std = {np_logits.std()}",
        )
      if self.temperature > 0:
        probs = torch.softmax(logits / self.temperature, dim=-1)
        next_token = sample_top_p(probs, self.top_p)
      else:
        next_token = torch.argmax(logits, dim=-1)
      next_token = next_token.reshape(-1)
      tokens[:, cur_pos] = next_token

      if next_token[0] in self.stop_tokens:
        break

    return tokens[:, :cur_pos + 1]
