from concurrent.futures import ProcessPoolExecutor
from threading import Lock
from typing import Any, cast

import tiktoken

_tokenizer: Any = None
_lock = Lock()
_executor = ProcessPoolExecutor(max_workers=1)


class GPT2Tokenizer:
    @staticmethod
    def _get_num_tokens_by_gpt2(text: str) -> int:
        """
        use gpt2 tokenizer to get num tokens
        """
        _tokenizer = GPT2Tokenizer.get_encoder()
        tokens = _tokenizer.encode(text, verbose=False)
        return len(tokens)

    @staticmethod
    def get_num_tokens(text: str) -> int:
        # return GPT2Tokenizer._get_num_tokens_by_gpt2(text)
        # Because this process needs more cpu resource, we turn this back before we find a better way to handle it.
        future = _executor.submit(GPT2Tokenizer._get_num_tokens_by_gpt2, text)
        result = future.result()
        return cast(int, result)

    @staticmethod
    def get_encoder() -> Any:
        global _tokenizer, _lock
        with _lock:
            if _tokenizer is None:
                # base_path = abspath(__file__)
                # gpt2_tokenizer_path = join(dirname(base_path), "gpt2")
                # _tokenizer = TransformerGPT2Tokenizer.from_pretrained(gpt2_tokenizer_path)

                # Try to use tiktoken to get the tokenizer because it is faster
                _tokenizer = tiktoken.get_encoding("gpt2")

            return _tokenizer
