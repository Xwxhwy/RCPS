import inspect
from typing import List, Dict

from RCPS_Project.rcps import constants


class LDLTokenizer:
    """
    A dynamic tokenizer for Layout Description Language (LDL).
    Builds its vocabulary directly from the constants.py module.
    """

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.special_tokens = {
            "pad": "<PAD>",
            "unk": "<UNK>",
        }
        self._build_vocab_from_constants()
        print(f"LDLTokenizer initialized. Vocabulary size: {self.get_vocab_size()}")

    def _build_vocab_from_constants(self):
        # Dynamically discover all uppercase string constants
        all_tokens = list(self.special_tokens.values())
        for name, value in inspect.getmembers(constants):
            if isinstance(value, str) and name.isupper():
                all_tokens.append(value)

        # Ensure consistent vocabulary order with special tokens first
        unique_tokens = sorted(list(set(all_tokens) - set(self.special_tokens.values())))
        final_token_list = list(self.special_tokens.values()) + unique_tokens

        self.word2idx = {token: i for i, token in enumerate(final_token_list)}
        self.idx2word = {i: token for token, i in self.word2idx.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.strip().split(' ')

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes an LDL string into a list of token IDs.

        Args:
            text (str): The LDL string to encode.
            add_special_tokens (bool): Whether to prepend SOS and append EOS tokens.

        Returns:
            List[int]: A list of token IDs.
        """
        tokens = self._tokenize(text)
        if add_special_tokens:
            tokens = [constants.SOS_TOKEN] + tokens + [constants.EOS_TOKEN]
        return [self.word2idx.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into an LDL string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.
            skip_special_tokens (bool): If True, special tokens are removed from the output.

        Returns:
            str: The decoded LDL string.
        """
        tokens = []
        special_ids_to_skip = {self.pad_token_id, self.sos_token_id, self.eos_token_id}
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids_to_skip:
                continue
            tokens.append(self.idx2word.get(token_id, self.unk_token))
        return ' '.join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def pad_token(self) -> str:
        return self.special_tokens['pad']

    @property
    def unk_token(self) -> str:
        return self.special_tokens['unk']

    @property
    def sos_token(self) -> str:
        return constants.SOS_TOKEN

    @property
    def eos_token(self) -> str:
        return constants.EOS_TOKEN

    @property
    def pad_token_id(self) -> int:
        return self.word2idx[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.word2idx[self.unk_token]

    @property
    def sos_token_id(self) -> int:
        return self.word2idx[self.sos_token]

    @property
    def eos_token_id(self) -> int:
        return self.word2idx[self.eos_token]