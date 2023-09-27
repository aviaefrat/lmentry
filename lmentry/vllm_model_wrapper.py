from typing import List, Union

import tvm
from vllm import LLM
from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)
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
        output_len: int,
        tokenizer
    ):
        self.tensor_parallel_size = 1
        self.seed = 0
        self.trust_remote_code = True

        self.llm = LLM(
            model=model_name,
            tokenizer=tokenizer,
            tensor_parallel_size=self.tensor_parallel_size,
            seed=self.seed,
            trust_remote_code=self.trust_remote_code,
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
        output_len: int,
        tokenizer):
        return VllmModelWrapper(model_name, config, output_len, tokenizer)

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
