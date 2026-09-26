"""Model interface supporting multiple backends."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    stop: Optional[List[str]] = None


class BaseModelBackend(ABC):
    """Abstract interface for any LLM backend."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        ...

    @abstractmethod
    def generate_batch(
        self, batch_messages: List[List[Message]], config: GenerationConfig
    ) -> List[str]:
        ...


class TransformersBackend(BaseModelBackend):
    """Hugging Face Transformers backend (with optional PEFT/LoRA)."""

    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False, **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quant_config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.load_in_4bit else None,
        )
        self._model.eval()

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        import torch

        prompt = self._tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def generate_batch(
        self, batch_messages: List[List[Message]], config: GenerationConfig
    ) -> List[str]:
        return [self.generate(msgs, config) for msgs in batch_messages]


class OpenAICompatibleBackend(BaseModelBackend):
    """Works with vLLM, Ollama, LM Studio, Together, Fireworks, etc."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        import httpx
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def generate(self, messages: List[Message], config: GenerationConfig) -> str:
        payload = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }
        if config.stop:
            payload["stop"] = config.stop

        resp = self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate_batch(
        self, batch_messages: List[List[Message]], config: GenerationConfig
    ) -> List[str]:
        return [self.generate(msgs, config) for msgs in batch_messages]


def create_model(config: Dict[str, Any]) -> BaseModelBackend:
    """Factory that creates the right backend from config."""
    backend = config.get("backend", "transformers").lower()
    name = config["name"]

    if backend == "transformers":
        return TransformersBackend(
            name,
            device=config.get("device", "auto"),
            load_in_4bit=config.get("load_in_4bit", False),
        )
    elif backend in ("vllm", "ollama", "openai", "openai_compatible"):
        return OpenAICompatibleBackend(
            name,
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key", os.getenv("OPENAI_API_KEY", "EMPTY")),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
