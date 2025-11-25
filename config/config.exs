import Config

# Taken from nanochat repo (https://github.com/karpathy/nanochat)
sequence_length = 1024
vocab_size = 65_536

config :nano_ai, NanoAi.LLM.Models.GPT,
  sequence_length: sequence_length,
  # Must match tokenizer vocab size
  vocab_size: vocab_size,
  num_layers: 12,
  num_heads: 6,
  # MQA heads for Multi-Query Attention
  # num_key_value_heads: 6,
  num_embed: 768,
  ffn_norm: :pre_norm,
  ffn_expand_factor: 4,
  ffn_type: :gelu,
  dropout_rate: 0.1,
  use_mix_precision: true,
  mix_precision_dtype: :bf16

config :nano_ai, NanoAi.Tokenizer,
  vocab_size: vocab_size,
  sequence_length: sequence_length,
  special_tokens: [
    "<|pad|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|eos|>"
  ]

config :nx, default_backend: {EMLX.Backend, device: :gpu}
