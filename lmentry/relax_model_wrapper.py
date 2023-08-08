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
  def __init__(self, model: Callable, tokenizer, args: argparse.Namespace):
    self.model = model
    self.tokenizer = tokenizer
    self.args = args

  def generate(
    self,
    prompt: str,
    max_gen_len: int,
    temperature: float = 1.1,
    top_p: float = 0.7,
    stream_interval: int = 2,
    stop_str: str = None,
    stop_tokens=None,
    keep_first_token=True,
  ):
    prompt_tokens = self.tokenizer.encode(prompt)
    stop_tokens = (
      [self.tokenizer.eos_token_id] if stop_tokens is None else stop_tokens
    )
    if not keep_first_token:
      prompt_tokens = prompt_tokens[1:]
    total_len = max_gen_len + len(prompt_tokens)
    tokens = torch.full((1, total_len), 0).to(torch.int32)
    tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens)
    start_pos = len(prompt_tokens)
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
      if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
      else:
        next_token = torch.argmax(logits, dim=-1)
      next_token = next_token.reshape(-1)
      tokens[:, cur_pos] = next_token
      # the following code assumes bsz == 1
      if next_token[0] in stop_tokens:
        stopped = True
      else:
        stopped = False

      i = cur_pos - start_pos
      if i % stream_interval == 0 or i == max_gen_len - 1 or stopped:
          output = tokens[0, : cur_pos + 1]
          output = self.tokenizer.decode(output, skip_special_tokens=True)
          if stop_str:
              pos = output.rfind(stop_str, len(prompt))
              if pos != -1:
                  output = output[:pos]
                  stopped = True
          yield output
      if stopped:
        break