# model_server.py

import gc
import logging
from typing import Optional
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# =========================
# Model Configuration
# =========================
class ModelConfig:
    """Configuration for the language model (shared by base & LoRA)."""

    # --- Paths ---
    REPO_ROOT = Path(__file__).resolve().parent
    BASE_MODEL_PATH = str(REPO_ROOT / "LLMs" / "Base_Model" / "Qwen3-14B")
    LORA_WEIGHTS_PATH = str(REPO_ROOT / "LLMs" / "LoRA_Weight" / "PE-Sleuth-Qwen3-14B-LoRA")

    # --- Runtime ---
    # NOTE: Should not exceed model's supported context window; your chunking should be below this.
    CTX_LENGTH: int = 32 * 1024
    MAX_NEW_TOKENS: int = 512

    # Quantization: 4, 8, or None
    QUANTIZATION_BITS: Optional[int] = 4

    # Optimizations
    USE_FLASH_ATTENTION: bool = True

    # Whether to enable "thinking" mode in chat template (model-dependent)
    ENABLE_THINKING: bool = False

    # Device selection
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Internal Utilities
# =========================
def _get_quantization_config(bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config."""
    if bits == 4:
        logging.info("Using 4-bit quantization configuration (nf4).")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    if bits == 8:
        logging.info("Using 8-bit quantization configuration.")
        return BitsAndBytesConfig(load_in_8bit=True)
    logging.info("Not using BitsAndBytes quantization.")
    return None


def _load_tokenizer(cfg: ModelConfig):
    logging.info(f"Loading tokenizer from {cfg.BASE_MODEL_PATH}...")
    tok = AutoTokenizer.from_pretrained(
        cfg.BASE_MODEL_PATH,
        trust_remote_code=True,
        model_max_length=cfg.CTX_LENGTH,  # ensure tokenizer knows max length
    )
    return tok


def _load_base_model_only(cfg: ModelConfig):
    """Load base model with YaRN & context-length adjustments."""
    if cfg.DEVICE == "cpu":
        raise RuntimeError("This model is intended for CUDA environments.")

    logging.info(f"Loading configuration and enabling YaRN to support {cfg.CTX_LENGTH} context.")
    tcfg = AutoConfig.from_pretrained(cfg.BASE_MODEL_PATH, trust_remote_code=True)
    # Explicitly enable YaRN and set max positions if the model supports it
    tcfg.use_yarn = True
    tcfg.max_position_embeddings = cfg.CTX_LENGTH

    quant_config = _get_quantization_config(cfg.QUANTIZATION_BITS)

    logging.info(f"Loading base model from {cfg.BASE_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL_PATH,
        config=tcfg,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_flash_attention_2=cfg.USE_FLASH_ATTENTION,
    )
    model.eval()
    return model


def _decode_chat_response(full_text: str, prompt_text: str) -> str:
    """
    Extract assistant reply from the generated text.
    Falls back gracefully if no special markers are found.
    """
    # Qwen-style "think" marker path (may vary by model/template)
    marker = "assistant\n<think>\n\n</think>\n\n"
    if marker in full_text:
        return full_text.split(marker)[-1].strip()

    # Generic fallback: drop the prompt prefix
    return full_text[len(prompt_text):].strip()


# =========================
# Shared Generation Mixin
# =========================
class _GenerationMixin:
    """
    Mixin providing generate() and get_tokenizer() assuming subclasses
    define: self.model, self.tokenizer, self.config
    """

    def generate(self, prompt: str) -> str:
        """
        Generate model response for a single-turn user message.
        """
        if getattr(self, "model", None) is None or getattr(self, "tokenizer", None) is None:
            raise RuntimeError("Model is not loaded, cannot generate text.")

        # Build input using the model's chat template
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.ENABLE_THINKING,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                use_cache=True,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=pad_id,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _decode_chat_response(full_response, prompt_text)

    def get_tokenizer(self):
        """Return the loaded tokenizer instance."""
        return getattr(self, "tokenizer", None)


# =========================
# Base Model Interface (kept)
# =========================
class LLMInterface(_GenerationMixin):
    """
    Interface for the BASE model (no LoRA applied).
    This preserves the original class name for backward compatibility.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize and load the base model and tokenizer according to config.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            logging.info("Base model already loaded; skip reloading.")
            return
        logging.info(f"Loading model on device: {self.config.DEVICE}")
        self.tokenizer = _load_tokenizer(self.config)
        self.model = _load_base_model_only(self.config)
        logging.info("Base model loaded and set to evaluation mode.")

    # --- New: explicit unload method to free VRAM ---
    def unload(self):
        """
        Unload the base model and tokenizer to free GPU memory.
        Safe to call multiple times.
        """
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
        finally:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Base model resources released.")


# =========================
# LoRA Model Interface (new)
# =========================
class LoRALLMInterface(_GenerationMixin):
    """
    Interface for the LoRA-adapted model.
    Loads base weights + applies LoRA adapters. Can be loaded, used, and unloaded
    independently from the base-only interface to save VRAM.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize and load the LoRA model and tokenizer according to config.

        Requirements:
            - config.LORA_WEIGHTS_PATH must be set to a valid adapter path.
        """
        if not config.LORA_WEIGHTS_PATH:
            raise ValueError("LORA_WEIGHTS_PATH must be set in ModelConfig to use LoRALLMInterface.")
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            logging.info("LoRA model already loaded; skip reloading.")
            return
        logging.info(f"Loading (LoRA) model on device: {self.config.DEVICE}")
        self.tokenizer = _load_tokenizer(self.config)
        base_model = _load_base_model_only(self.config)

        logging.info(f"Applying LoRA adapters from {self.config.LORA_WEIGHTS_PATH} ...")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.config.LORA_WEIGHTS_PATH,
            device_map="auto",
        )
        self.model.eval()
        logging.info("LoRA-adapted model loaded and set to evaluation mode.")

    # --- New: explicit unload method to free VRAM ---
    def unload(self):
        """
        Unload the LoRA model (and underlying base) and tokenizer to free GPU memory.
        Safe to call multiple times.
        """
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
        finally:
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("LoRA model resources released.")
