from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5EncoderModel

DEFAULT_GTR_MODEL_NAME = "sentence-transformers/gtr-t5-base"
MAX_LENGTH = 128


def resolve_torch_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if prefer_gpu and mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    unmasked_outputs = hidden_states * attention_mask[..., None]
    return unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]


@dataclass
class GTREncoder:
    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module
    device: str
    max_length: int = MAX_LENGTH

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        all_embeddings: list[torch.Tensor] = []
        self.model.eval()

        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            batches = tqdm(batches, desc="GTR encode", unit="batch")

        with torch.no_grad():
            for start in batches:
                batch = texts[start : start + batch_size]
                tokenizer_call: Any = getattr(self.tokenizer, "__call__")
                inputs = tokenizer_call(
                    batch,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                ).to(self.device)
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )
                embeddings = mean_pool(outputs.last_hidden_state, inputs.attention_mask)
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)

        stacked_embeddings = torch.cat(all_embeddings, dim=0)
        if convert_to_numpy:
            return stacked_embeddings.cpu().numpy().astype(np.float32, copy=False)
        if convert_to_tensor:
            return stacked_embeddings
        return stacked_embeddings.cpu().tolist()


def load_gtr_encoder(
    model_name: str = DEFAULT_GTR_MODEL_NAME,
    *,
    prefer_gpu: bool = True,
) -> tuple[GTREncoder, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = resolve_torch_device(prefer_gpu=prefer_gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    return GTREncoder(tokenizer=tokenizer, model=model, device=device), device

