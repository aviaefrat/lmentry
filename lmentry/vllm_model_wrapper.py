import os
import random
from typing import List, Tuple, Union

import torch
import numpy as np

import tvm
from vllm import LLM, SamplingParams
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, AutoModelForCausalLM, PreTrainedTokenizerBase)
from tqdm import tqdm
import json

from vllm.logger import init_logger

logger = init_logger(__name__)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

class VllmModelWrapper:
    def __init__(self,
        model_name: str,
        config: dict,
        tensor_parallel_size: int,
        seed: int,
        trust_remote_code: bool,
        use_beam_search: bool,
        output_len: int,
        n: int,
        tokenizer
    ):
        print("model", model_name)
        print("tokenizer", tokenizer)
        self.llm = LLM(
            model=model_name,
            tokenizer=tokenizer,  # TODO: temporary solution
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )
    
        self.device = tvm.device("cuda")
        self.model = self.llm
    
    def generate(self,
        in_prompts: List[str],
        sampling_params
    ):  
        for prompt in in_prompts:
            self.model._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
            )

        outputs = self.model._run_engine(use_tqdm=True)
        vllm_outputs = []
        for output in outputs:
            prompt = output.prompt
            print("prompt",prompt)
            generated_text = output.outputs[0].text
            print("generated_text",generated_text)
            vllm_outputs.append(generated_text)
        return vllm_outputs
    

    @staticmethod
    def get_vllm_model(
        model_name: str,
        config: dict,
        tensor_parallel_size: int,
        seed: int,
        trust_remote_code: bool,
        use_beam_search: bool,
        output_len: int,
        n: int,
        tokenizer):
        return VllmModelWrapper(model_name, config, tensor_parallel_size, seed, trust_remote_code, use_beam_search, output_len, n, tokenizer)

    @staticmethod
    def get_vllm_tokenizer(
        tokenizer_name: str,
        *args,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        **kwargs,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Gets a tokenizer for the given model name via Huggingface."""
        if tokenizer_mode == "slow":
            if kwargs.get("use_fast", False):
                raise ValueError(
                    "Cannot use the fast tokenizer in slow tokenizer mode.")
            kwargs["use_fast"] = False

        if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
            logger.info(
                "For some LLaMA-based models, initializing the fast tokenizer may "
                "take a long time. To eliminate the initialization time, consider "
                f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
                "tokenizer.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                **kwargs)
        except TypeError as e:
            # The LLaMA tokenizer causes a protobuf error in some environments.
            err_msg = (
                "Failed to load the tokenizer. If you are using a LLaMA-based "
                f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
                "tokenizer.")
            raise RuntimeError(err_msg) from e
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported, suggest using the --trust-remote-code flag.
            if (not trust_remote_code and
                ("does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e))):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer is a custom "
                    "tokenizer not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead.")
        return tokenizer

    @staticmethod
    def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[Tuple[str, int, int]]:
        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data for data in dataset
            if len(data["conversations"]) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]

        # Tokenize the prompts and completions.
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 or output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            filtered_dataset.append((prompt, prompt_len, output_len))

        # Sample the requests.
        sampled_requests = random.sample(filtered_dataset, num_requests)
        return sampled_requests