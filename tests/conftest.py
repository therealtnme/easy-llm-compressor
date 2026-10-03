"""Offline test fixtures: tiny real Transformers models + a tiny local tokenizer.

No network access is required. Models are constructed from real HF configs so the
module graph, config semantics and save/load paths are genuinely exercised.
"""
from __future__ import annotations

import os
from typing import Callable

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------- #
# tiny config factories
# --------------------------------------------------------------------------- #
def _cfg(name: str):
    if name == "llama":
        from transformers import LlamaConfig

        return LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attn_implementation="eager",
            tie_word_embeddings=False,
        )
    if name == "qwen2":
        from transformers import Qwen2Config

        return Qwen2Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attn_implementation="eager",
            tie_word_embeddings=True,
        )
    if name == "mistral":
        from transformers import MistralConfig

        return MistralConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=64, attn_implementation="eager",
            tie_word_embeddings=False,
        )
    if name == "gpt2":
        from transformers import GPT2Config

        return GPT2Config(
            vocab_size=128, n_embd=32, n_layer=3, n_head=4, n_inner=64,
            n_positions=64, attn_implementation="eager",
        )
    if name == "opt":
        from transformers import OPTConfig

        return OPTConfig(
            vocab_size=128, hidden_size=32, ffn_dim=64, num_hidden_layers=3,
            num_attention_heads=4, max_position_embeddings=64,
            attn_implementation="eager",
        )
    if name == "bert":
        from transformers import BertConfig

        return BertConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=64,
            attn_implementation="eager",
        )
    if name == "t5":
        from transformers import T5Config

        return T5Config(
            vocab_size=128, d_model=32, d_ff=64, num_layers=3, num_decoder_layers=3,
            num_heads=4, d_kv=8, decoder_start_token_id=0, attn_implementation="eager",
        )
    raise ValueError(name)


_MODEL_CLS = {
    "llama": ("AutoModelForCausalLM",),
    "qwen2": ("AutoModelForCausalLM",),
    "mistral": ("AutoModelForCausalLM",),
    "gpt2": ("AutoModelForCausalLM",),
    "opt": ("AutoModelForCausalLM",),
    "bert": ("AutoModel",),
    "t5": ("AutoModelForSeq2SeqLM",),
}


def build_tiny(name: str):
    """Instantiate a tiny, randomly-initialised model of the given family."""
    import transformers

    cfg = _cfg(name)
    cls_name = _MODEL_CLS[name][0]
    cls = getattr(transformers, cls_name)
    torch.manual_seed(0)
    model = cls.from_config(cfg)
    model.eval()
    return model, cfg


def save_tiny(name: str, path: str):
    model, cfg = build_tiny(name)
    model.save_pretrained(path)
    make_tokenizer().save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_models_dir(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("tiny_models")
    for name in ("llama", "qwen2", "gpt2", "opt", "bert", "t5", "mistral"):
        save_tiny(name, str(root / name))
    return str(root)


@pytest.fixture
def model_factory() -> Callable[[str], object]:
    return build_tiny


# --------------------------------------------------------------------------- #
# tiny offline tokenizer (no downloads)
# --------------------------------------------------------------------------- #
_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a", "cat",
    "sat", "on", "mat", "hello", "world", "language", "model", "is", "small",
    "token", "data", "neuron", "layer", "prune", "compress", "test", "and",
    "of", "to", "in", "with", "for", "that", "this", "it", "was", "are", "be",
]


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return make_tokenizer()


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for w in _WORDS:
        if w not in vocab:
            vocab[w] = len(vocab)
    # single characters as a fallback so arbitrary text is encodable
    for c in "abcdefghijklmnopqrstuvwxyz0123456789.,-":
        if c not in vocab:
            vocab[c] = len(vocab)

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=64,
    )


@pytest.fixture(scope="session")
def tiny_texts() -> list[str]:
    return [
        "the quick brown fox jumps over the lazy dog",
        "a cat sat on the mat and the dog was lazy",
        "language model compression reduces neuron count and layer depth",
        "the token data is small and the model is a test model",
        "prune the neuron and compress the layer with the token data",
        "hello world this is a small language model test",
        "the fox was quick and the dog was lazy on the mat",
        "compress the model to reduce the neuron count in the layer",
    ]
