defmodule NanoAi.LLM.Models.GPT do
  @moduledoc """
  GPT-style decoder-only transformer language model.

  ## Overview

  This module implements a causal language model similar to GPT-2/GPT-3, designed for
  next-token prediction. The model reads a sequence of tokens and predicts what comes next,
  enabling autoregressive text generation.

  ## Architecture

  The model follows a standard transformer decoder architecture:

      Input Token IDs: [batch, seq_len]
        ↓
      Token Embeddings: [batch, seq_len, num_embed]
        +
      Position Embeddings: [batch, seq_len, num_embed]
        ↓
      Transformer Block 1 (Attention + FFN)
        ↓
      Transformer Block 2
        ↓
      ...
        ↓
      Transformer Block N (num_layers)
        ↓
      Final Layer Norm
        ↓
      Output Projection → Logits: [batch, seq_len, vocab_size]

  ## Key Components

  ### 1. Embeddings Layer

  Converts token IDs into dense vector representations:

  - **Token Embeddings**: Maps each token ID (0-65535) to a 768-dimensional vector
  - **Position Embeddings**: Adds positional information so the model knows token order
  - Both embeddings are learned during training and summed together

  **Why position embeddings?**
  Without them, "The cat sat" would be identical to "sat cat The" - just a bag of tokens.
  Position embeddings encode that "The" is at position 0, "cat" at position 1, etc.

  ### 2. Transformer Blocks

  The core of the model, stacked 12 times. Each block contains:

  **Attention Layer**: Allows tokens to gather information from previous tokens
  - Uses causal masking so token i can only see tokens 0..i (not future tokens)
  - Multi-head attention runs 6 parallel attention operations
  - Each head can learn different patterns (syntax, semantics, etc.)

  **Feed-Forward Network (FFN)**: Processes each token independently
  - Expands to 4× the embedding size (768 → 3072)
  - Applies GELU non-linearity
  - Projects back to original size (3072 → 768)
  - This is where the model stores most of its "knowledge"

  **Residual Connections & Layer Norm**: Stabilize training
  - Allows gradients to flow easily through 12 layers
  - Normalizes activations to prevent instability

  ### 3. Output Head

  Projects the final representations back to vocabulary space:
  - Maps 768-dimensional vectors to 65,536 logits (one per vocabulary token)
  - No bias term (common practice in modern LLMs)
  - Logits are unnormalized scores that get converted to probabilities

  ## Configuration

  Default configuration (from config.exs):

  ```elixir
  vocab_size: 65_536          # Tokenizer vocabulary size
  sequence_length: 1_024      # Maximum context window
  num_layers: 12              # Transformer blocks
  num_heads: 6                # Attention heads per block
  num_embed: 768              # Embedding/hidden dimension
  ffn_expand_factor: 4        # FFN internal size = 768 × 4 = 3072
  ffn_norm: :pre_norm         # Layer norm before sub-layers (GPT-3 style)
  ffn_type: :gelu             # Activation function
  ```

  ## Model Size

  Approximate parameter counts:

  - Token embeddings: 65,536 × 768 = **50.3M**
  - Position embeddings: 1,024 × 768 = **0.8M**
  - 12 Transformer blocks: ~7M each = **84M**
  - Output projection: 768 × 65,536 = **50.3M**
  - **Total: ~185M parameters**

  ## Training Objective

  Causal language modeling (next-token prediction):

  ```elixir
  Input:  [token_1, token_2, token_3, ..., token_n]
  Target: [token_2, token_3, token_4, ..., token_n+1]
  ```

  The model learns to predict each token given all previous tokens.
  At each position i, it predicts token i+1 using only tokens 0..i (enforced by causal masking).

  ## Data Flow Example

  ```elixir
  # 1. Input: A batch of 8 sequences, each 1024 tokens
  input_ids = Nx.tensor([[1, 234, 5678, ...], ...])  # [8, 1024]

  # 2. Embeddings: Convert IDs to vectors
  embeddings = token_emb + position_emb  # [8, 1024, 768]

  # 3. Transformer blocks: Refine representations 12 times
  # Each token gathers context and processes information
  hidden = transformer_blocks(embeddings)  # [8, 1024, 768]

  # 4. Output: Project to vocabulary
  logits = output_head(hidden)  # [8, 1024, 65536]

  # 5. Loss: Compare predictions with targets
  # logits[i] predicts token[i+1]
  loss = cross_entropy(logits[:, :-1], targets[:, 1:])
  ```

  ## Inference (Text Generation)

  During generation, the model runs autoregressively:

  ```elixir
  # Start with prompt
  tokens = tokenize("The cat")  # [42, 156]

  # Generate next tokens one at a time
  loop do
    logits = model(tokens)           # [1, seq_len, 65536]
    next_token = sample(logits[-1])  # Pick from distribution
    tokens = tokens ++ [next_token]  # Append to sequence
    break if next_token == eos_token
  end

  # Result: "The cat sat on the mat"
  ```

  ## Usage

  ```elixir
  # Build the model
  model = NanoAi.LLM.Models.GPT.build()

  # Compile for execution
  {init_fn, predict_fn} = Axon.build(model)

  # Initialize parameters
  params = init_fn.(Nx.template({1, 1024}, :s64), %{})

  # Forward pass
  input_ids = Nx.tensor([[1, 2, 3, 4, 5]])  # [1, 5]
  logits = predict_fn.(params, input_ids)   # [1, 5, 65536]

  # Training
  train_data = NanoAi.LLM.Dataloader.from_files("data/*.txt")
  trained_params = NanoAi.LLM.Trainer.train(model, train_data, epochs: 10)
  ```

  ## Pre-Norm vs Post-Norm

  This model uses **pre-norm** (layer norm before attention/FFN):

  ```
  x = x + Attention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))
  ```

  Pre-norm is more stable for deep networks and is used in GPT-3, LLaMA, and most modern LLMs.

  Post-norm (original transformer) applies normalization after the residual:
  ```
  x = LayerNorm(x + Attention(x))
  x = LayerNorm(x + FFN(x))
  ```

  ## Design Decisions

  - **No bias in output projection**: Reduces parameters without hurting performance
  - **GELU activation**: Smooth, probabilistic ReLU variant used in GPT-2/3
  - **Causal attention**: Essential for autoregressive generation
  - **Learned positions**: Simpler than sinusoidal, works well for fixed context
  - **Pre-norm**: More stable training for 12+ layers

  ## References

  - GPT-2 Paper: "Language Models are Unsupervised Multitask Learners"
  - GPT-3 Paper: "Language Models are Few-Shot Learners"
  - Attention Paper: "Attention is All You Need"
  """

  import Nx.Defn

  alias NanoAi.LLM.Layers.Embedding
  alias NanoAi.LLM.Layers.Transformer
  alias NanoAi.Tokenizer
  alias Nx.Serving

  @config Application.compile_env(:nano_ai, __MODULE__)

  @spec build(opts :: keyword()) :: Axon.t()
  def build(opts \\ @config) do
    opts =
      Keyword.validate!(opts, [
        :vocab_size,
        :sequence_length,
        :num_layers,
        :num_heads,
        :num_embed,
        :ffn_expand_factor,
        :ffn_norm,
        :ffn_type,
        dropout_rate: 0.0,
        use_mix_precision: false,
        mix_precision_dtype: :bf16
      ])

    opts
    |> embeddings()
    |> transformer_blocks(opts)
    |> final_norm()
    |> output_projection(opts)
    |> then(fn model ->
      if opts[:use_mix_precision] do
        Axon.MixedPrecision.apply_policy(
          model,
          Axon.MixedPrecision.create_policy(
            params: :f32,
            compute: opts[:mix_precision_dtype],
            output: :f32
          ),
          except: [:layer_norm]
        )
      else
        model
      end
    end)
  end

  @doc """
  Creates an Nx.Serving for next token generation.

  ## Parameters
    - model: Axon model
    - params: Trained model parameters
    - opts: Generation options

  ## Options
    - :temperature - Sampling temperature. Default: 1.0
  """
  def serving(params \\ %{}, opts \\ []) do
    # Build the model once
    {_init_fn, predict_fn} = Axon.build(build(), mode: :inference)

    fn _defn_opts ->
      fn input_ids ->
        predict_fn.(params, input_ids)
      end
    end
    |> Serving.new()
    |> Serving.client_preprocessing(&preprocess/1)
    |> Serving.client_postprocessing(&postprocess/2)
  end

  defp preprocess({input, temperature}) do
    with {:ok, %{ids: input_ids}} <- Tokenizer.encode(input) do
      input_ids_tensor = Nx.from_binary(input_ids, :u32)
      {Nx.Batch.stack([input_ids_tensor]), {temperature}}
    end
  end

  defp postprocess({output, _}, {temperature}) do
    output
    |> apply_temperature(temperature)
    |> sample_from_logits(System.unique_integer())
    |> then(fn token_id_tensor ->
      case Tokenizer.decode(token_id_tensor) do
        {:ok, token} -> token
        {:error, _} -> ""
      end
    end)
  end

  defnp apply_temperature(logits, temperature) do
    if temperature == 1.0 do
      logits
    else
      logits / temperature
    end
  end

  defnp sample_from_logits(logits, seed) do
    logits = logits[0][-1]
    probs = stable_softmax(logits)
    key = Nx.Random.key(Nx.as_type(seed, :s64))
    {sampled, _key} = Nx.Random.choice(key, Nx.iota(Nx.shape(logits)), probs, samples: 1)
    sampled[0]
  end

  defnp softmax(logits) do
    logits |> Nx.exp() |> then(&(&1 / Nx.sum(&1)))
  end

  defnp stable_softmax(logits) do
    logits |> Nx.reduce_max() |> then(&Nx.subtract(logits, &1)) |> softmax()
  end

  defp embeddings(opts) do
    embedding_opts = [
      vocab_size: opts[:vocab_size],
      num_embed: opts[:num_embed],
      sequence_length: opts[:sequence_length]
    ]

    Embedding.positional(embedding_opts)
  end

  # Stacks multiple transformer blocks to build the deep network.
  #
  # Each block refines the token representations by allowing tokens to:
  # 1. Gather information from other tokens (via attention)
  # 2. Process that information independently (via FFN)
  #
  # Blocks are stacked sequentially, with the output of one block feeding into the next.
  # This creates increasingly abstract representations:
  # - Early blocks: Learn basic patterns (syntax, word order)
  # - Middle blocks: Learn compositional features (phrases, relations)
  # - Late blocks: Learn high-level semantics (meaning, reasoning)
  #
  # Parameters per block (for default config):
  # - Attention: ~2.4M parameters (Q, K, V, output projections)
  # - FFN: ~4.7M parameters (up/down projections)
  # - Layer norms: ~3K parameters (negligible)
  # - Total: ~7.1M per block × 12 blocks = ~85M parameters
  #
  # The model uses either pre-norm or post-norm architecture:
  # - Pre-norm (default): More stable, used in GPT-3 and modern LLMs
  # - Post-norm: Original transformer, slightly better performance but harder to train
  defp transformer_blocks(input, opts) do
    Enum.reduce(1..opts[:num_layers], input, fn _, input ->
      case opts[:ffn_norm] do
        :pre_norm ->
          Transformer.pre_norm(input, opts[:num_embed], opts[:num_heads],
            ffn_type: opts[:ffn_type],
            expand_factor: opts[:ffn_expand_factor],
            dropout_rate: opts[:dropout_rate]
          )

        :post_norm ->
          Transformer.post_norm(input, opts[:num_embed], opts[:num_heads],
            ffn_type: opts[:ffn_type],
            expand_factor: opts[:ffn_expand_factor],
            dropout_rate: opts[:dropout_rate]
          )
      end
    end)
  end

  # Applies layer normalization after all transformer blocks.
  #
  # This final normalization ensures the representations are well-behaved before
  # projecting to vocabulary space. It normalizes across the embedding dimension,
  # making the model's outputs more stable.
  #
  # Layer norm formula: (x - mean) / sqrt(variance + eps) * gamma + beta
  # - Computes mean and variance across the num_embed dimension (768)
  # - gamma (scale) and beta (shift) are learned parameters
  # - Helps prevent the output projection from receiving extreme values
  defp final_norm(input) do
    Axon.layer_norm(input)
  end

  # Projects the final hidden states to vocabulary logits.
  #
  # This layer transforms the 768-dimensional representations into 65,536 logits
  # (one score per vocabulary token). The logits represent unnormalized probabilities
  # and will be converted to a distribution via softmax during training/inference.
  #
  # Parameters: 768 × 65,536 = 50,331,648 (~50M parameters)
  #
  # No bias is used (use_bias: false) because:
  # 1. Saves parameters without hurting performance
  # 2. Common practice in modern LLMs (GPT-3, LLaMA, etc.)
  # 3. Bias is redundant when using layer norm before this layer
  #
  # Shape transformation:
  # - Input:  [batch, seq_len, 768]
  # - Output: [batch, seq_len, 65536]
  #
  # During inference, logits at position i predict the next token (position i+1)
  defp output_projection(input, opts) do
    Axon.dense(input, opts[:vocab_size], use_bias: false)
  end
end
