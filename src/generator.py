from __future__ import annotations

import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenGenerator:
    """低显存文本生成：优先 4bit 1.5B，失败则回退 0.5B fp16。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_4bit: bool = True,
        fallback_model: Optional[str] = None,
        fallback_use_4bit: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        try:
            self._load(model_name, use_4bit)
        except Exception as exc:
            if not fallback_model:
                raise
            print(f"[generator] 主模型加载失败 ({exc})，回退到 {fallback_model}")
            self._unload()
            self.model_name = fallback_model
            self._load(fallback_model, fallback_use_4bit)

    def _load(self, model_name: str, use_4bit: bool) -> None:
        load_kwargs: dict = {"trust_remote_code": True}
        if self.cache_dir:
            load_kwargs["cache_dir"] = self.cache_dir

        print(f"[generator] 加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        kwargs: dict = {"trust_remote_code": True}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if use_4bit and self.device == "cuda" and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            kwargs["device_map"] = "auto" if self.device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if kwargs.get("device_map") is None and self.device == "cuda":
            self.model = self.model.to("cuda")

    def _unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        assert self.model is not None and self.tokenizer is not None
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
